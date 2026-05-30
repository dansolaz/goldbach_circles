# Manual: Run the Prime Rolling Circles Animation

## 1) Create or reuse virtual environment

If `.venv` already exists, you can skip creation.

```bash
python3 -m venv .venv
```

## 2) Activate the virtual environment

```bash
source .venv/bin/activate
```

## 3) Install dependencies

```bash
pip install -r requirements.txt
```

## 4) Run the animation

```bash
python prime_rolling_circles.py
```

## 5) Optional configuration

Edit the configuration section at the top of `prime_rolling_circles.py`:

- `MAX_NUMBER`
- `PRIME_MODE` and `MANUAL_CIRCLES`
- `MARK_FROM`, `START_ROLLING_FROM`
- `SHOW_*` flags
- `EXPORT_ANIMATION`, `EXPORT_FORMAT`

## 6) Optional export

Set:

- `EXPORT_ANIMATION = True`
- `EXPORT_FORMAT = "mp4"` or `"gif"`

Then run again:

```bash
python prime_rolling_circles.py
```

For MP4 export, you may need `ffmpeg` installed on your system.
