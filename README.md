# IdeaForge-GeoLoc-InterIIT15


UAV visual geo-localisation for the ideaForge problem statement (Inter-IIT Tech Meet 15.0). Code is the section-by-section conversion of `GeoLoc_Pipeline1.ipynb`.

```
python run.py                      # whole run on the development photos
python run.py --n-photos 60        # smaller sample
python run.py --no-minima          # correlation only
python run.py --dino --tests       # more switches: --sweep --sensitivity --independence
```

Layout: `src/<section>/` (each with its own README), `dataset/` (tracked), `dataset_temp/` (never uploaded). Settings are in `src/header/settings.py`. **Where the big data files go (maps, model weights): `dataset/README.md`.**
