# Streamlit patient-style research demo

This repository includes a patient-style Streamlit research demo:

```bash
streamlit run app/streamlit_app.py
```

## What the demo does

The app allows a user to enter routine clinical assessment values:

- age
- sex
- education
- MMSE
- CDR global
- CDR sum of boxes
- FAQ total
- optional ADAS13
- number of available visits

It then displays 1/2/3/5-year research-demo risk estimates.

## Important boundary

The public demo estimate is **not** the private frozen model output. It does not load model checkpoints, feature tensors, raw data, preprocessing objects, calibration objects, or patient-level predictions.

The purpose is to show the intended interface style for a future controlled research demonstration.

## Not for clinical use

This is not a clinical tool, not medical advice, and not a deployed medical device.
