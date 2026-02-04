# 🔧 Troubleshooting Guide

## Common Issues and Solutions

---

## 1. Checkpoint Loading Issues

### ❌ Error: "File not found: checkpoint_partX.safetensors"

**Cause**: Checkpoint not uploaded or wrong path

**Solutions**:
1. **Verify upload location**:
   ```python
   import os
   print(os.listdir('/kaggle/input/'))
   print(os.listdir('/kaggle/input/model-checkpoint/'))
   ```
   
2. **Upload to correct path**:
   - Must be: `/kaggle/input/model-checkpoint/checkpoint_partX.safetensors`
   - Not: `/kaggle/working/` or `/kaggle/input/`

3. **Check filename exactly**:
   - Correct: `checkpoint_part1.safetensors`
   - Wrong: `checkpoint_part_1.safetensors`, `checkpoint1.safetensors`

4. **Re-upload if needed**:
   - Go to Kaggle "Add Data" → "Upload"
   - Create dataset named `model-checkpoint`
   - Upload the file

---

### ❌ Error: "Checkpoint size mismatch" or "Shape mismatch"

**Cause**: Wrong checkpoint for current part or corrupted file

**Solutions**:
1. **Verify you have the correct part**:
   - Part 2 needs `checkpoint_part1`
   - Part 3 needs `checkpoint_part2`
   - Part 4 needs `checkpoint_part3`
   - Part 5 needs `checkpoint_final`

2. **Check file size**:
   ```bash
   ls -lh /kaggle/input/model-checkpoint/
   ```
   Should be ~1.2 GB

3. **Re-download and re-upload**:
   - Download might have been interrupted
   - Try uploading again

4. **Verify file integrity**:
   ```python
   from safetensors.torch import load_file
   checkpoint = load_file('/kaggle/input/model-checkpoint/checkpoint_part1.safetensors')
   print(f"Keys: {len(checkpoint.keys())}")  # Should be ~100+
   ```

---

## 2. Memory Issues

### ❌ Error: "CUDA out of memory"

**Cause**: GPU memory exhausted

**Solutions**:
1. **Already optimized** (batch_size=1 is minimum)
   
2. **Restart kernel**:
   - Click "Restart & Run All"
   - Clears cached memory

3. **Try T4 GPU instead of P100**:
   - Sometimes T4 has better memory management
   - Settings → Accelerator → T4 GPU

4. **Reduce gradient accumulation** (last resort):
   ```python
   # In training arguments cell, change:
   gradient_accumulation_steps=4  # from 8
   ```
   Note: This reduces effective batch size

5. **Emergency: Disable gradient checkpointing**:
   ```python
   gradient_checkpointing=False
   ```
   ⚠️ This will use MORE memory initially but might work with specific configurations

---

### ❌ Error: "RuntimeError: DataLoader worker terminated unexpectedly"

**Cause**: Worker process ran out of memory

**Solutions**:
1. **Reduce number of workers**:
   ```python
   # In training arguments:
   dataloader_num_workers=1  # from 2
   ```

2. **Restart kernel and try again**

---

## 3. Training Issues

### ❌ Warning: "Loss is NaN" or "Loss exploding"

**Cause**: Numerical instability

**Solutions**:
1. **Check checkpoint loaded correctly**:
   - Verify correct part number
   - Verify TRAINING_PART matches uploaded checkpoint

2. **Reduce learning rate** (emergency fix):
   ```python
   learning_rate=1e-4  # from 5e-4
   ```

3. **Enable gradient clipping**:
   ```python
   # Add to TrainingArguments:
   max_grad_norm=1.0
   ```

4. **Verify FP16 is enabled**:
   ```python
   fp16=True
   ```

---

### ❌ Issue: "WER not decreasing"

**Cause**: Training not progressing

**Diagnostics**:
1. **Check training loss**:
   - Should decrease steadily
   - If flat, model not learning

2. **Verify data loading**:
   ```python
   # Add a test cell:
   sample = dataset['train'][0]
   print(f"Audio shape: {sample['audio'].shape}")
   print(f"Text: {sample['text']}")
   ```

3. **Check learning rate**:
   - Part 1-4: Should be 5e-4
   - Part 4 Phase 2: Should be 5e-6

**Solutions**:
1. **Verify preprocessing working**:
   - Check audio files are loading
   - Check transcripts are not empty

2. **Check vocabulary size**:
   ```python
   print(f"Vocab size: {len(processor.tokenizer)}")
   ```
   Should be ~100-150 for Bengali

3. **Continue training longer**:
   - WER might decrease slowly at first

---

### ❌ Issue: "Training too slow"

**Cause**: Dataset size or GPU throttling

**Solutions**:
1. **Check GPU is actually being used**:
   ```python
   print(torch.cuda.is_available())
   print(model.device)
   ```

2. **Monitor GPU usage**:
   ```python
   !nvidia-smi
   ```

3. **Reduce logging frequency** (minor speedup):
   ```python
   logging_steps=200  # from 100
   ```

4. **Accept slower speed**:
   - Long audio clips are slow by nature
   - batch_size=1 is necessary

---

## 4. Download/Upload Issues

### ❌ Error: "Download failed" or "Connection timeout"

**Cause**: Large file size or connection issue

**Solutions**:
1. **Use Kaggle's download button**:
   - Navigate to file in Kaggle interface
   - Right-click → Download

2. **Download via Kaggle API** (advanced):
   ```bash
   kaggle kernels output [your-kernel-name] -p /output/path
   ```

3. **Split download if timeout**:
   - For very large files, download in parts
   - Not usually needed for 1.2 GB files

---

### ❌ Error: "Upload failed" or "File too large"

**Cause**: File size or upload limit

**Solutions**:
1. **Verify file size**:
   - Checkpoints should be ~1.2 GB
   - Kaggle limit is 20 GB (we're well under)

2. **Use dataset upload instead of direct upload**:
   - Kaggle "Add Data" → "Upload" → "New Dataset"
   - More reliable than drag-and-drop

3. **Check internet connection**:
   - Stable connection needed for large uploads
   - Consider using wired connection

---

## 5. Configuration Issues

### ❌ Error: "TRAINING_PART not defined"

**Cause**: Configuration cell not run

**Solutions**:
1. **Run configuration cell**:
   - Find cell with `TRAINING_PART = 1`
   - Execute it before other cells

2. **Use "Run All"**:
   - Ensures cells run in order

---

### ❌ Issue: "Wrong part running"

**Cause**: TRAINING_PART not set correctly

**Verification**:
```python
print(f"Current part: {TRAINING_PART}")
print(f"Config: {current_config['description']}")
```

**Solutions**:
1. **Change TRAINING_PART value**:
   ```python
   TRAINING_PART = 2  # Change this number
   ```

2. **Restart kernel and run again**:
   - Ensures clean state

---

## 6. Inference Issues

### ❌ Error: "Test files not found"

**Cause**: Dataset path incorrect

**Solutions**:
1. **Verify dataset attached**:
   ```python
   print(os.listdir('/kaggle/input/'))
   ```

2. **Check test path**:
   ```python
   TEST_AUDIO_PATH = "/kaggle/input/dl-sprint-4-0-bengali-long-form-speech-recognition/transcription/transcription/test"
   print(os.listdir(TEST_AUDIO_PATH))
   ```

3. **Adjust path if needed**:
   - Dataset structure might be different
   - Update path in configuration cell

---

### ❌ Issue: "Transcriptions are empty or wrong"

**Cause**: Model or post-processing issue

**Solutions**:
1. **Verify final checkpoint loaded**:
   ```python
   print(f"Loading: {current_config['load_checkpoint']}")
   ```

2. **Test on single sample**:
   ```python
   sample = test_data[0]
   result = transcribe_audio(sample['audio'], model, processor)
   print(f"Raw: {result}")
   print(f"Post-processed: {postprocess_text(result)}")
   ```

3. **Check post-processing working**:
   ```python
   test_text = "আমি বাংলায় গান গাই"
   print(postprocess_text(test_text))
   # Should end with । (Danda)
   ```

---

## 7. Session Timeout Issues

### ❌ Issue: "Session expired before completion"

**Cause**: Training taking longer than 12 hours

**Prevention**:
1. **Monitor time regularly**:
   - Check remaining time in Kaggle UI
   - Each part should be well under 12 hours

2. **If approaching limit**:
   ```python
   # Manually save emergency checkpoint:
   from safetensors.torch import save_file
   save_file(model.state_dict(), '/kaggle/working/emergency_checkpoint.safetensors')
   ```

3. **Download immediately**:
   - Don't wait for training to "complete"
   - Download partial checkpoint

4. **Resume in next session**:
   - Upload emergency checkpoint
   - Adjust epochs to continue

---

## 8. Processor Issues

### ❌ Error: "Processor not found" or "Vocabulary error"

**Cause**: Processor not saved/loaded correctly

**Solutions**:
1. **Processor auto-generated** from vocabulary:
   - Should be created automatically in data prep phase

2. **Verify vocabulary file**:
   ```python
   vocab_path = f"{OUTPUT_DIR}/vocab.json"
   with open(vocab_path, 'r') as f:
       vocab = json.load(f)
   print(f"Vocab size: {len(vocab)}")
   ```

3. **Recreate processor if needed**:
   - Re-run vocabulary and processor creation cells

---

## 9. Post-Processing Issues

### ❌ Error: "bnUnicodeNormalizer not found"

**Cause**: Library not installed

**Solutions**:
```python
!pip install bnunicodenormalizer
```

---

### ❌ Issue: "Text not ending with Danda (।)"

**Cause**: Post-processing not applied

**Verification**:
```python
# Check function is defined:
result = postprocess_text("test")
print(result)  # Should end with ।
```

---

## 10. Logging and Monitoring

### ❌ Issue: "Can't see training progress"

**Cause**: Logging not visible or redirected

**Solutions**:
1. **Check logging configuration**:
   ```python
   logging_steps=100  # Log every 100 steps
   ```

2. **View TensorBoard logs**:
   ```python
   %load_ext tensorboard
   %tensorboard --logdir={LOGS_DIR}
   ```

3. **Manual progress check**:
   ```python
   # Add in training loop:
   print(f"Step: {trainer.state.global_step}")
   print(f"Loss: {trainer.state.log_history[-1]}")
   ```

---

## Emergency Contacts

If none of these solutions work:

1. **Check Kaggle Status**: https://www.kaggle.com/status
2. **Kaggle Forums**: https://www.kaggle.com/discussions
3. **Review notebook output**: Often contains specific error messages
4. **Start fresh**: Sometimes cleanest solution is new session

---

## Debug Mode

To enable detailed debugging, add at top of notebook:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Verbose error messages
import warnings
warnings.filterwarnings('default')

# Check all imports
print("✓ All imports successful")
```

---

## Quick Fixes Summary

| Issue | Quick Fix |
|-------|-----------|
| Out of memory | Restart kernel |
| File not found | Check upload path |
| NaN loss | Verify checkpoint loaded |
| Slow training | Normal for long audio |
| WER not improving | Continue training longer |
| Session timeout | Save emergency checkpoint |
| Vocabulary error | Re-run vocab cells |

---

Still stuck? Review the error message carefully - it usually tells you exactly what's wrong! 🔍
