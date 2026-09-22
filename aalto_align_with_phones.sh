#!/bin/bash
# Zundanen diagnostic variant of Lingsoft's Aalto ELG align.sh.
# It keeps the normal word-level result and appends phone-level intervals
# to the same per-utterance result file using a __PHONE__: prefix.

csv_file=$1
debugBoolean=$2
textDirBoolean=$3
src_for_wav=$4
src_for_txt=$5

project_dir=/opt/kaldi/egs/kohdistus
src_for_mdl=/opt/kaldi/egs/align

cd "$project_dir"

bin_folder=/opt/kaldi/egs/align/aligning_with_Docker/bin

ln -s ../wsj/s5/utils utils
ln -s ../wsj/s5/steps steps
ln -s "$src_for_mdl"/conf conf

mkdir exp
mkdir -p data/align

ln -s "$src_for_mdl"/exp/nnet3 exp/nnet3

cp ../wsj/s5/path.sh .

sed -i '1 s/...$//' path.sh
mkdir data/dict
mkdir data/lang

cat <<EOF2 > data/dict/optional_silence.txt
SIL
EOF2

cat <<EOF2 > data/dict/silence_phones.txt
NSN
SIL
SPN
EOF2

cat <<EOF2 > data/dict/nonsilence_phones.txt
2
A
I
N
U
b
d
e
f
g
h
j
k
l
m
n
o
p
r
s
t
v
y
{
EOF2

if [ "$textDirBoolean" = "textDirTrue" ]
then
  python $bin_folder/make_wav_and_utt2spk.py "$project_dir" "$src_for_wav" --txtpath "$src_for_txt"
else
  python $bin_folder/make_wav_and_utt2spk.py "$project_dir" "$src_for_wav"
fi
sed -i 's/!sil/!SIL/g' data/align/text
sed -i 's/<unk>/<UNK>/g' data/align/text

cut -f 2- -d ' ' data/align/text > corpus

python $bin_folder/change_lex_pho.py corpus $bin_folder/$csv_file
mv lexicon.txt data/dict

extra=3
utils/prepare_lang.sh --num-extra-phone-disambig-syms $extra data/dict "<UNK>" data/lang/local data/lang

mkdir -p exp/align
ln -s exp/nnet3/chain/tree exp/align/tree
ln -s exp/nnet3/chain/final.mdl exp/align/final.mdl

utils/fix_data_dir.sh data/align
utils/copy_data_dir.sh data/align data/align_hires

nj=1

steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf --cmd "run.pl" data/align_hires
steps/compute_cmvn_stats.sh data/align_hires

utils/fix_data_dir.sh data/align_hires

steps/online/nnet2/extract_ivectors_online.sh --cmd "run.pl" --nj $nj data/align_hires exp/nnet3/extractor exp/align/ivectors_hires
steps/nnet3/align.sh --nj 1 --use_gpu false --online_ivector_dir exp/align/ivectors_hires data/align_hires/ data/lang/ exp/nnet3/chain/ exp/align_ali

# The chain model uses frame subsampling.  The alignment vectors contain one
# item per subsampled frame, so CTM conversion must scale the normal 10 ms
# feature frame shift by the model's frame_subsampling_factor (3 for FI).
frame_subsampling_factor=1
if [ -f exp/align_ali/frame_subsampling_factor ]; then
  frame_subsampling_factor=$(cat exp/align_ali/frame_subsampling_factor)
elif [ -f exp/nnet3/chain/frame_subsampling_factor ]; then
  frame_subsampling_factor=$(cat exp/nnet3/chain/frame_subsampling_factor)
fi
frame_shift=$(python -c "print(0.01 * int('${frame_subsampling_factor}'))")

# Existing ELG word-level result, but with the correct frame shift.
steps/get_train_ctm.sh --frame-shift "$frame_shift" data/align_hires data/lang exp/align_ali
cp exp/align_ali/ct* .
python $bin_folder/ctm2results.py ctm

# Additional phone-level result for Zundanen.
# ali-to-phones CTM columns: utterance, channel, start, duration, phone-id.
/opt/kaldi/src/bin/ali-to-phones --ctm-output --frame-shift="$frame_shift" exp/nnet3/chain/final.mdl \
  "ark:gunzip -c exp/align_ali/ali.1.gz|" - \
  | utils/int2sym.pl -f 5 data/lang/phones.txt > phone.ctm

python - <<'PY'
from pathlib import Path

phone_ctm = Path("phone.ctm")
if phone_ctm.exists():
    for line in phone_ctm.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        utt, _channel, start_text, duration_text, phone = parts[:5]
        try:
            start = float(start_text)
            duration = float(duration_text)
        except ValueError:
            continue
        end = start + max(0.0, duration)
        # Lingsoft's ctm2results.py has already created <utterance>.ctm
        # with three columns: start end word. Append phone rows using a
        # marker that Zundanen strips on the host side.
        out = Path(f"{utt}.ctm")
        with out.open("a", encoding="utf-8") as fp:
            fp.write(f"{start:.4f} {end:.4f} __PHONE__:{phone}\n")
PY

if [ "$debugBoolean" = "true" ]
then
  cp data/align/text .
  cp data/align/wav.scp .
  cp data/dict/lexicon.txt .
fi
rm corpus path.sh conf steps utils
rm -r exp data
