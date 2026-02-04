# Bengali Speech Recognition - Fine-Tuning Documentation

This directory contains everything needed to train a wav2vec2 model for Bengali speech recognition on Kaggle.

---

## 📚 Documentation Files

### 1. **fine-tune.ipynb** ⭐
The main Jupyter notebook containing the complete training pipeline.
- Multi-part training support (5 parts)
- Checkpoint saving/loading with safetensors
- Complete preprocessing, training, and inference code
- Ready to run on Kaggle

### 2. **TRAINING_GUIDE.md** 📖
Comprehensive step-by-step guide for running all 5 parts.
- Detailed instructions for each part
- File management guidelines
- Troubleshooting section
- Expected performance metrics

### 3. **QUICK_REFERENCE.md** ⚡
Quick reference card with essential information.
- Part-by-part configuration snippets
- Common commands
- File paths
- Error quick fixes
- Print this and keep it handy!

### 4. **VISUAL_WORKFLOW.md** 📊
Visual diagrams and flowcharts.
- Complete training pipeline diagram
- File flow visualization
- Time budget breakdown
- WER progress chart
- Memory footprint analysis

### 5. **TROUBLESHOOTING.md** 🔧
Comprehensive troubleshooting guide.
- Common errors and solutions
- Memory issue fixes
- Upload/download problems
- Configuration issues
- Debug mode instructions

---

## 🚀 Quick Start

1. **Read first**: TRAINING_GUIDE.md (10 minutes)
2. **Open notebook**: fine-tune.ipynb in Kaggle
3. **Set configuration**: `TRAINING_PART = 1`
4. **Run**: Execute all cells
5. **Repeat**: Follow guide for parts 2-5

---

## 📋 Training Overview

### Multi-Part Training Strategy

Due to Kaggle's 12-hour session limit, training is split into 5 parts:

| Part | Duration | Description | Output |
|------|----------|-------------|--------|
| 1 | ~3 hrs | Phase 1: Epochs 1-18 | `checkpoint_part1.safetensors` |
| 2 | ~3 hrs | Phase 1: Epochs 19-36 | `checkpoint_part2.safetensors` |
| 3 | ~3 hrs | Phase 1: Epochs 37-54 | `checkpoint_part3.safetensors` |
| 4 | ~4 hrs | Phase 1: Epochs 55-70 + Phase 2 (all) | `checkpoint_final.safetensors` |
| 5 | ~1 hr | Inference only | `submission.csv` |

**Total**: ~14 hours split across 5 Kaggle sessions

---

## 📁 File Structure

```
fine-tune/
├── fine-tune.ipynb           # Main notebook (RUN THIS)
├── README.md                 # This file
├── TRAINING_GUIDE.md         # Detailed guide
├── QUICK_REFERENCE.md        # Quick reference card
├── VISUAL_WORKFLOW.md        # Visual diagrams
└── TROUBLESHOOTING.md        # Problem solving
```

---

## 🎯 Expected Results

### Performance Metrics
- **Final WER**: ~0.4-0.5 (40-50% word error rate)
- **Training Loss**: Decreases from ~3.0 → ~0.7
- **Total Parameters**: ~315M (wav2vec2-large-xlsr-53)

### Output Files
- **Checkpoints**: 4 files, each ~1.2 GB
- **Final submission**: CSV with Bengali transcriptions
- **Training logs**: TensorBoard compatible

---

## 🔑 Key Features

### Paper-Faithful Implementation
✅ Based on: *Applying wav2vec2 for Speech Recognition on Bengali Common Voices Dataset* (Shahgir et al., arXiv:2209.06581)

✅ Exact specifications:
- Model: facebook/wav2vec2-large-xlsr-53
- Preprocessing: 16kHz, silence removal, 1-10s clips
- Phase 1: 70 epochs, LR=5e-4
- Phase 2: 7 epochs, LR=5e-6 (exposure boost)
- Post-processing: Bengali normalization + Danda (।)

### Kaggle Optimizations
✅ Multi-part checkpoint system
✅ Safetensors format (efficient + safe)
✅ Memory-efficient (batch_size=1, FP16, gradient checkpointing)
✅ Clear progress tracking
✅ Automatic session management

---

## ⚙️ System Requirements

### Kaggle Environment
- **GPU**: P100 or T4 (16 GB VRAM)
- **RAM**: 16 GB system memory
- **Storage**: ~5 GB for checkpoints
- **Time**: 5 sessions × ~3-4 hours each

### Local Requirements (for checkpoint storage)
- **Storage**: ~6 GB free space
- **Internet**: Stable connection for uploads/downloads

---

## 📖 Reading Order

**For first-time users:**
1. Start with: **TRAINING_GUIDE.md** (comprehensive)
2. Keep handy: **QUICK_REFERENCE.md** (during training)
3. If stuck: **TROUBLESHOOTING.md** (problem solving)
4. For visualization: **VISUAL_WORKFLOW.md** (optional)

**For experienced users:**
1. **QUICK_REFERENCE.md** only
2. Run **fine-tune.ipynb** with appropriate `TRAINING_PART`

---

## 🎓 Paper Reference

This implementation reproduces the methodology from:

> **Applying wav2vec2 for Speech Recognition on Bengali Common Voices Dataset**  
> Shahgir et al.  
> arXiv:2209.06581  
> https://arxiv.org/abs/2209.06581

### Key Findings from Paper
- wav2vec2-large-xlsr-53 performs best for Bengali ASR
- Two-phase training improves vocabulary exposure
- Post-processing (normalization + Danda) reduces error
- WER of ~0.4-0.5 achievable on Bengali Common Voice

---

## 🔄 Workflow Summary

```
Part 1 → Download checkpoint → Upload to new session → Part 2
  ↓                                                       ↓
  └────────────────────────────────────────────────────→ Part 3
                                                           ↓
                                                         Part 4
                                                           ↓
                                                         Part 5
                                                           ↓
                                                    submission.csv
```

Each part:
1. Set `TRAINING_PART` configuration
2. Run all cells in notebook
3. Download checkpoint file
4. Start new session for next part

---

## ✅ Checklist

Before starting:
- [ ] Read TRAINING_GUIDE.md
- [ ] Have stable internet connection
- [ ] Have ~6 GB free storage locally
- [ ] Kaggle account with GPU access

For each part:
- [ ] Fresh Kaggle session started
- [ ] GPU enabled
- [ ] Previous checkpoint uploaded (if not Part 1)
- [ ] TRAINING_PART configured correctly
- [ ] Run all cells
- [ ] Download checkpoint before session expires

After completion:
- [ ] All 5 parts completed
- [ ] submission.csv downloaded
- [ ] Transcriptions verified (Bengali text + Danda)
- [ ] Ready to submit!

---

## 📞 Support

### If you encounter issues:

1. **Check**: TROUBLESHOOTING.md (covers 90% of issues)
2. **Verify**: Configuration and file paths
3. **Review**: Notebook output for error messages
4. **Search**: Kaggle discussions for similar issues

### Common Issues:
- **File not found**: Check upload path
- **Out of memory**: Restart kernel
- **Session timeout**: Download checkpoint before 11 hours
- **Wrong checkpoint**: Verify part number

---

## 🎉 Success Criteria

You'll know you're successful when:
- ✅ All 5 parts complete without errors
- ✅ WER decreases steadily (→ ~0.4-0.5)
- ✅ submission.csv contains Bengali text
- ✅ All transcriptions end with । (Danda)
- ✅ No corrupted or empty predictions

---

## 📊 Monitoring Progress

### During Training:
- Watch training loss (should decrease)
- Check validation WER (should decrease)
- Monitor time remaining on Kaggle
- View TensorBoard logs (optional)

### After Each Part:
- Verify checkpoint file size (~1.2 GB)
- Check metrics plot (loss/WER curves)
- Confirm successful download

---

## 🚦 Status Indicators

The notebook shows clear status messages:

```
✓  Success indicator (green)
⚠️  Warning indicator (yellow)
❌  Error indicator (red)
📥  Download required
⏭️  Skipped (not applicable for current part)
```

---

## 🏆 Final Output

After completing all 5 parts, you'll have:

1. **Trained Model**: Full wav2vec2 Bengali ASR model
2. **Predictions**: submission.csv with test set transcriptions
3. **Logs**: Complete training history
4. **Metrics**: WER and loss curves
5. **Checkpoints**: All intermediate states (optional to keep)

---

## 🎯 Next Steps After Completion

1. **Verify submission.csv format**:
   - Columns: filename, transcription
   - Encoding: UTF-8
   - All rows present

2. **Optional improvements**:
   - Add language model decoding
   - Fine-tune post-processing
   - Ensemble with other models

3. **Submit results**:
   - Follow competition guidelines
   - Submit submission.csv

---

## 📝 Version Information

- **Notebook Version**: 1.0
- **Paper**: arXiv:2209.06581
- **Model**: facebook/wav2vec2-large-xlsr-53
- **Framework**: Transformers (Hugging Face)
- **Platform**: Kaggle (P100/T4 GPU)

---

## 📜 License

Implementation based on research paper methodology. 
Model weights: As per Hugging Face model license.
Code: Educational/Research use.

---

**Ready to start? Open `fine-tune.ipynb` in Kaggle and begin with Part 1!** 🚀

For detailed instructions, see **TRAINING_GUIDE.md**
