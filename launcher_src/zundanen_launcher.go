//go:build windows

package main

import (
    "os"
    "os/exec"
    "path/filepath"
    "syscall"
    "unsafe"
)

const (
    appTitle       = "Zundanen"
    createNoWindow = 0x08000000
)

func utf16Ptr(s string) *uint16 {
    p, _ := syscall.UTF16PtrFromString(s)
    return p
}

func messageBox(text string) {
    user32 := syscall.NewLazyDLL("user32.dll")
    proc := user32.NewProc("MessageBoxW")
    proc.Call(
        0,
        uintptr(unsafe.Pointer(utf16Ptr(text))),
        uintptr(unsafe.Pointer(utf16Ptr(appTitle))),
        uintptr(0x10), // MB_ICONERROR
    )
}

func main() {
    exePath, err := os.Executable()
    if err != nil {
        messageBox("Could not determine the application folder.")
        return
    }

    root := filepath.Dir(exePath)
    pyw := filepath.Join(root, "runtime", "tts", ".venv", "Scripts", "pythonw.exe")
    launcher := filepath.Join(root, "desktop_launcher.py")

    if _, err := os.Stat(pyw); err != nil {
        messageBox("Local runtime was not found.\n\nRun setup.bat first.")
        return
    }
    if _, err := os.Stat(launcher); err != nil {
        messageBox("desktop_launcher.py was not found.\n\nReinstall or restore the Zundanen files.")
        return
    }

    cmd := exec.Command(pyw, launcher)
    cmd.Dir = root
    cmd.SysProcAttr = &syscall.SysProcAttr{
        HideWindow:     true,
        CreationFlags: createNoWindow,
    }

    if err := cmd.Start(); err != nil {
        messageBox("Zundanen could not be started.\n\n" + err.Error())
        return
    }
}
