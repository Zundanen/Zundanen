Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
try {
    [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
    $OutputEncoding = [Console]::OutputEncoding
} catch {}
$ProgressPreference = "SilentlyContinue"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runtime = Join-Path $Root "runtime"
$RvcPython = Join-Path $Runtime "rvc\.venv\Scripts\python.exe"
$StatusPath = Join-Path $Runtime "accent_support_status.json"
$AlignerImage = "lingsoft/aalto-kaldi-align:5.1.1-elg"
$script:LastNativeExitCode = 0

function Write-Status([string]$State, [string]$Message, [bool]$RebootRequired = $false) {
    New-Item -ItemType Directory -Force -Path $Runtime | Out-Null
    @{
        state = $State
        message = $Message
        reboot_required = $RebootRequired
        updated_at = (Get-Date -Format o)
    } | ConvertTo-Json | Set-Content -Path $StatusPath -Encoding UTF8
}
function Refresh-Path {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user"
}
function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $p = New-Object Security.Principal.WindowsPrincipal($id)
    return $p.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}
function Find-DockerDesktop {
    $candidates = @(
        (Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"),
        (Join-Path $env:ProgramFiles "Docker\Docker Desktop.exe")
    )
    if ($env:LOCALAPPDATA) {
        $candidates += @(
            (Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop\Docker Desktop.exe"),
            (Join-Path $env:LOCALAPPDATA "Programs\Docker\Docker\Docker Desktop.exe"),
            (Join-Path $env:LOCALAPPDATA "Docker\Docker Desktop.exe")
        )
    }
    foreach ($p in $candidates) { if (Test-Path $p) { return $p } }
    return $null
}
function Find-DockerCli {
    $cmd = Get-Command docker.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $desktop = Find-DockerDesktop
    $candidates = @()
    if ($desktop) {
        $parent = Split-Path -Parent $desktop
        $candidates += @(
            (Join-Path $parent "resources\bin\docker.exe"),
            (Join-Path $parent "resources\docker.exe")
        )
    }
    $candidates += @(
        (Join-Path $env:ProgramFiles "Docker\Docker\resources\bin\docker.exe"),
        (Join-Path $env:ProgramFiles "Docker\Docker\resources\docker.exe")
    )
    if ($env:LOCALAPPDATA) {
        $candidates += @(
            (Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop\resources\bin\docker.exe"),
            (Join-Path $env:LOCALAPPDATA "Programs\Docker\Docker\resources\bin\docker.exe")
        )
    }
    foreach ($p in $candidates) { if (Test-Path $p) { return $p } }
    return $null
}

# Windows PowerShell 5.1 can turn stderr from native executables into a terminating
# NativeCommandError when $ErrorActionPreference is "Stop".  Accent setup relies on
# non-zero native exit codes for normal "not installed / not ready yet" checks, so run
# native executables with Continue temporarily and record their exit code explicitly.
function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [string[]]$Arguments = @(),

        [switch]$Quiet
    )

    $oldPreference = $ErrorActionPreference
    $script:LastNativeExitCode = -1
    $ErrorActionPreference = "Continue"

    try {
        if ($Quiet) {
            & $FilePath @Arguments *> $null
        } else {
            # Merge stderr into stdout and write it to the host so it does not become
            # function output. This keeps diagnostics visible without contaminating the
            # exit-code handling below.
            & $FilePath @Arguments 2>&1 | ForEach-Object { Write-Host $_ }
        }
        $script:LastNativeExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $oldPreference
    }
}

if (-not (Test-Admin)) {
    Write-Host "Requesting administrator permission..." -ForegroundColor Yellow
    $args = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ('"' + $PSCommandPath + '"'))
    Start-Process powershell.exe -Verb RunAs -ArgumentList $args
    exit 0
}

try {
    Write-Host "Zundanen - Accent support setup" -ForegroundColor Green
    Write-Host "Only missing Accent components will be installed."
    Write-Status "starting" "Checking Accent support components."

    if (-not (Test-Path $RvcPython)) {
        throw "The normal Zundanen environment was not found. Run setup.bat first, then click Install Accent support again."
    }

    Write-Host ""
    Write-Host "[1/4] PyWORLD" -ForegroundColor Cyan

    # Do not intentionally trigger ImportError/Traceback just to test installation.
    # find_spec returns exit code 1 cleanly when PyWORLD is missing.
    Invoke-NativeCommand `
        -FilePath $RvcPython `
        -Arguments @(
            "-c",
            "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('pyworld') else 1)"
        ) `
        -Quiet

    if ($script:LastNativeExitCode -ne 0) {
        Write-Status "installing_pyworld" "Installing PyWORLD."
        Invoke-NativeCommand `
            -FilePath $RvcPython `
            -Arguments @(
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "pyworld==0.3.5"
            )

        if ($script:LastNativeExitCode -ne 0) {
            throw "PyWORLD installation failed (exit code $($script:LastNativeExitCode))."
        }
    }

    Invoke-NativeCommand `
        -FilePath $RvcPython `
        -Arguments @(
            "-c",
            "import pyworld; print('[OK] PyWORLD', pyworld.__version__)"
        )

    if ($script:LastNativeExitCode -ne 0) {
        throw "PyWORLD could not be imported (exit code $($script:LastNativeExitCode))."
    }

    Write-Host ""
    Write-Host "[2/4] WSL2" -ForegroundColor Cyan
    $wsl = Join-Path $env:WINDIR "System32\wsl.exe"
    $wslReady = $false

    if (Test-Path $wsl) {
        Invoke-NativeCommand -FilePath $wsl -Arguments @("--status") -Quiet
        $wslReady = ($script:LastNativeExitCode -eq 0)
    }

    $rebootRequired = $false
    if (-not $wslReady) {
        if (-not (Test-Path $wsl)) {
            throw "wsl.exe was not found. WSL2 requires a supported Windows 10/11 installation."
        }

        Write-Status "installing_wsl" "Installing WSL2 Windows components."
        Write-Host "Installing WSL2 Windows components..." -ForegroundColor Yellow
        Invoke-NativeCommand -FilePath $wsl -Arguments @("--install", "--no-distribution")

        if ($script:LastNativeExitCode -ne 0) {
            throw "WSL2 installation failed (exit code $($script:LastNativeExitCode))."
        }
        $rebootRequired = $true
    } else {
        Write-Host "[OK] WSL2 is available." -ForegroundColor Green
    }

    # If WSL was just enabled, Windows may require a reboot before this command can work.
    # On an already-ready WSL installation, a failure here is non-fatal but should be visible.
    if (-not $rebootRequired) {
        Invoke-NativeCommand -FilePath $wsl -Arguments @("--set-default-version", "2") -Quiet
        if ($script:LastNativeExitCode -ne 0) {
            Write-Warning "Could not set the default WSL version to 2 (exit code $($script:LastNativeExitCode))."
        }
    }

    Write-Host ""
    Write-Host "[3/4] Docker Desktop" -ForegroundColor Cyan
    $dockerDesktop = Find-DockerDesktop
    if (-not $dockerDesktop) {
        Write-Status "installing_docker" "Installing Docker Desktop."
        $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
        if (-not $winget) {
            throw "WinGet was not found. Install/repair App Installer, then try again."
        }

        Write-Host "Installing Docker Desktop..." -ForegroundColor Yellow
        Invoke-NativeCommand `
            -FilePath $winget.Source `
            -Arguments @(
                "install",
                "--id", "Docker.DockerDesktop",
                "-e",
                "--source", "winget",
                "--accept-package-agreements",
                "--accept-source-agreements",
                "--silent",
                "--disable-interactivity"
            )

        $wingetInstallExitCode = $script:LastNativeExitCode
        if ($wingetInstallExitCode -ne 0) {
            # Some WinGet versions return a non-zero code when the package is already
            # installed/up-to-date. Verify actual installation before treating it as fatal.
            Invoke-NativeCommand `
                -FilePath $winget.Source `
                -Arguments @(
                    "list",
                    "--id", "Docker.DockerDesktop",
                    "-e",
                    "--accept-source-agreements",
                    "--disable-interactivity"
                ) `
                -Quiet

            if ($script:LastNativeExitCode -ne 0) {
                throw "Docker Desktop installation failed (WinGet exit code $wingetInstallExitCode)."
            }
        }

        Refresh-Path
        $dockerDesktop = Find-DockerDesktop
    }

    if (-not $dockerDesktop) {
        throw "Docker Desktop was installed but could not be located. Restart Windows and try again."
    }
    Write-Host "[OK] Docker Desktop is installed." -ForegroundColor Green

    if ($rebootRequired) {
        Write-Status "reboot_required" "Windows restart required before Docker can be started." $true
        Write-Host ""
        Write-Host "Windows must be restarted to finish enabling WSL2." -ForegroundColor Yellow
        Write-Host "After restarting Windows, open Zundanen and click Install Accent support again." -ForegroundColor Yellow
        exit 0
    }

    Write-Host ""
    Write-Host "[4/4] Docker engine + Aalto image" -ForegroundColor Cyan
    $docker = Find-DockerCli
    if (-not $docker) {
        Refresh-Path
        $docker = Find-DockerCli
    }
    if (-not $docker) {
        throw "docker.exe was not found after Docker Desktop installation."
    }

    Invoke-NativeCommand -FilePath $docker -Arguments @("info", "--format", "{{.ServerVersion}}") -Quiet
    if ($script:LastNativeExitCode -ne 0) {
        Write-Status "starting_docker" "Starting Docker Desktop."
        Write-Host "Starting Docker Desktop..." -ForegroundColor Yellow
        Start-Process -FilePath $dockerDesktop | Out-Null

        $deadline = (Get-Date).AddMinutes(4)
        do {
            Start-Sleep -Seconds 3
            Invoke-NativeCommand -FilePath $docker -Arguments @("info", "--format", "{{.ServerVersion}}") -Quiet
            if ($script:LastNativeExitCode -eq 0) { break }
        } while ((Get-Date) -lt $deadline)

        if ($script:LastNativeExitCode -ne 0) {
            throw "Docker Desktop did not become ready. Complete any Docker Desktop first-run window, then click Install Accent support again."
        }
    }
    Write-Host "[OK] Docker engine is running." -ForegroundColor Green

    Invoke-NativeCommand -FilePath $docker -Arguments @("image", "inspect", $AlignerImage) -Quiet
    if ($script:LastNativeExitCode -ne 0) {
        Write-Status "pulling_image" "Downloading the Aalto alignment Docker image."
        Write-Host "Downloading Aalto alignment image..." -ForegroundColor Yellow
        Invoke-NativeCommand -FilePath $docker -Arguments @("pull", $AlignerImage)

        if ($script:LastNativeExitCode -ne 0) {
            throw "Could not download the Aalto alignment Docker image (exit code $($script:LastNativeExitCode))."
        }
    }
    Write-Host "[OK] Aalto alignment image is ready." -ForegroundColor Green

    Write-Status "complete" "Accent support installation completed."
    Write-Host ""
    Write-Host "ACCENT SUPPORT READY" -ForegroundColor Green
    Write-Host "Return to Zundanen. The Accent editor will appear automatically."
    exit 0
}
catch {
    $message = $_.Exception.Message
    Write-Status "error" $message $false
    Write-Host ""
    Write-Host "[ERROR] $message" -ForegroundColor Red
    exit 1
}
