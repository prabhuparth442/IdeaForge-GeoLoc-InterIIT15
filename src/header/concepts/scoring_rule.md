# The scoring rule: what "good" means, and when to say "not matched"

**Code:** `src/header/scoring.py` · **Used by:** all of `evaluation/`, and the design of the answer-or-skip decision in `result_thresholding/`.

**Read this first.** Every design choice in the project (how accurate we must be, why there is a second matching stage, why we sometimes skip a photo) comes from this one scoring rule. If you only read one concept note, read this one.

**What you need to know beforehand:** nothing beyond school geometry. All the maths here is similar triangles, one square root and one inequality.

---

## 1. The big picture in plain words

For every test photo the system returns either a position (latitude/longitude) or "Not Matched". The organisers then measure how far our position is from the truth and award points:

* very close → **+10**
* a bit off → +6 or +3
* far off → 0
* very far off → **−20** (a penalty)
* "Not Matched" → 0

So there are two things to learn from the rule:

1. **What counts as "close"?** It is not a fixed number of metres. It depends on how high the drone flew. This is the quantity σ (sigma), Sections 2 and 3.
2. **A wrong answer costs more than a right answer earns.** So it is sometimes better to say "Not Matched". Section 5 works out exactly when.

## 2. Step 1: from pixels to metres (GSD)

Everything in the rule is measured in metres on the ground, but the camera counts pixels. We need the conversion factor: *how many metres does one pixel cover?* This is called the **ground sample distance (GSD)**.

Picture the camera as a pinhole looking straight down from height `h`:

```
              camera (pinhole)
                  *
                 /|\
          f     / | \            f  = focal length, in PIXELS
   image   ---+-----+---         u  = distance of the point in the image from the centre (pixels)
   plane   |<-u->|               h  = height above ground (metres)
                  | h            X  = distance of the ground point from the point right below the camera (metres)
   ground  ------nadir----X------
```

Two triangles share the same apex (the pinhole): a small one inside the camera (height `f`, base `u`) and a big one down to the ground (height `h`, base `X`). They are similar, so their sides have the same ratio:

$$\frac{u}{f}=\frac{X}{h}\quad\Longrightarrow\quad X=u\cdot\frac{h}{f}.$$

Set `u = 1` pixel. Then `X` is the ground covered by one pixel:

$$\boxed{\text{GSD}=\frac{h}{f}\ \text{metres per pixel}}$$

**Worked example.** The official camera has `f ≈ 200` pixels (see `cam_fx` in the metadata).

* At `h = 100 m`: GSD = 100/200 = **0.5 m per pixel**.
* At `h = 350 m`: GSD = 350/200 = **1.75 m per pixel**.

Higher means each pixel covers more ground, so everything on the ground looks smaller and blurrier in the picture.

## 3. Step 2: the allowed error σ

The organisers need a yardstick that grows with height, because a higher photo simply cannot be positioned as precisely. They build σ from two error sources:

* a **fixed 8 m**: the reference map and GPS themselves are not perfect, whatever the height;
* **3 pixels of ground size**, `3·GSD`: we are not expected to find a position more precisely than about three image pixels.

When two independent errors act together you do not add their sizes; you add their *squares* and take the root (this is the same rule that gives the length of the hypotenuse, or the standard deviation of a sum of independent quantities):

$$\sigma=\sqrt{8^{2}+(3\cdot\text{GSD})^{2}}\ \text{metres}.$$

**Worked example at 150 m** (`GSD = 0.75`): `3·GSD = 2.25`, so `σ = √(64 + 5.06) = √69.06 = 8.31 m`.

| height h | GSD = h/200 | σ |
|---|---|---|
| 100 m | 0.50 m | 8.14 m |
| 150 m | 0.75 m | 8.31 m |
| 200 m | 1.00 m | 8.54 m |
| 350 m | 1.75 m | 9.57 m |
| 1000 m | 5.00 m | 17.0 m (the problem statement's own example: "about 17 m") |

**What this tells us for the project.** On our 100–350 m development photos σ stays between about 8 and 10 m. The coarse search compares cells of 3.4 to 13.6 m (`image_processing/concepts/map_zoom_levels.md`). That is the same size as σ itself, so the coarse search alone can land on the right spot but cannot reliably be within one σ. That is the reason the second, finer stage exists.

## 4. Step 3: points for one photo

Let `e` be the distance in metres between our answer and the true position. Convert it to "how many σ" by dividing: `ratio = e / σ`. Then:

| ratio = e/σ | points | in words |
|---|---|---|
| ≤ 1 | **+10** | within one σ |
| ≤ 2 | +6 | |
| ≤ 4 | +3 | |
| ≤ 8 | 0 | |
| > 8 | **−20** | far off |
| (not matched) | 0 | skipped |

**Worked example, four photos at 100 m (σ = 8.14 m):**

| photo | error e | ratio | points |
|---|---|---|---|
| A | 6 m | 0.74 | +10 |
| B | 15 m | 1.84 | +6 |
| C | 30 m | 3.69 | +3 |
| D | 90 m | 11.06 | −20 |

Total = 10 + 6 + 3 − 20 = −1. Look at that: three good answers are wiped out by one bad one.

### From points to the final score

$$\text{score}=60\cdot\max\!\Big(\frac{\sum\text{points}}{10\cdot N},\ 0\Big).$$

Read it from the inside out:

* `10·N` is the best total possible (every one of the `N` photos earns +10).
* `Σ points / (10·N)` is therefore "what share of the maximum did we collect", a number from about −2 to 1.
* `max(·, 0)` stops the score from going below zero.
* `60 ·` stretches it onto a 0 to 60 scale.

Our four photos: `−1 / 40 = −0.025` → clipped to 0 → score **0**.

**The problem statement's example (check this by hand).** 100 photos: 40 earn +10, 20 earn +6, 2 earn −20, 38 earn 0.
`Σ = 400 + 120 − 40 + 0 = 480`; `480 / 1000 = 0.48`; `60 · 0.48 = 28.8`. The function `check_scoring()` prints this number, and your copy of the code must print 28.8. If it does not, something in `scoring.py` is wrong.

## 5. Step 4: when is it better to skip? (the 2/3 rule)

Suppose that, for a certain kind of photo, our answer would be good with probability `p` (earning `r` points) and bad with probability `1 − p` (losing 20 points). The *expected* points from answering are

$$E=p\cdot r-(1-p)\cdot 20.$$

Skipping gives exactly 0. So answering is the better choice when `E > 0`:

$$p\,r>20(1-p)\ \Longleftrightarrow\ p\,(r+20)>20\ \Longleftrightarrow\ \boxed{p>\frac{20}{20+r}}.$$

| a right answer earns r | we need to be right with probability p above |
|---|---|
| +10 | 20/30 = 2/3 ≈ 0.67 |
| +6 | 20/26 ≈ 0.77 |
| +3 | 20/23 ≈ 0.87 |

**Feel for it (r = 10).** Expected points per answered photo:

| our chance of being right | expected points |
|---|---|
| 0.60 | −2.0 (worse than skipping) |
| 0.67 | +0.1 (break-even) |
| 0.75 | +2.5 |
| 0.90 | +7.0 |

The famous "2/3" is the *best case*, where every correct answer earns the full +10. In reality some correct answers land in the +6 or +3 tier, so the true break-even is higher. This is why the project has a **confidence signal** (the number of matcher inliers, see `fine_matching/` and `result_thresholding/`): it estimates `p` for each photo, and we only answer when `p` looks high enough.

## 6. The code, line by line (`scoring.py`)

```python
def sigma_m(altitude_m, focal_px):
    assert altitude_m > 0 ...                       # input checks
    height = max(altitude_m, 100.0)                 # never use a height below 100 m
    gsd = height / focal_px                         # Section 2
    return math.sqrt(8.0 ** 2 + (3.0 * gsd) ** 2)   # Section 3
```

* `focal_px` must be the focal length **in pixels** (`camera.fx`). A focal length in millimetres would give nonsense by a factor of hundreds.
* `max(altitude_m, 100.0)` means a drone at 60 m is scored as if it were at 100 m.

```python
def points_for_error(error_m, altitude_m, focal_px):
    ratio = error_m / sigma_m(altitude_m, focal_px)
    if ratio <= 1:  return 10
    if ratio <= 2:  return 6
    if ratio <= 4:  return 3
    if ratio <= 8:  return 0
    return -20
```

The chain of `if`s runs from the best tier down; the first one that is true returns. The order matters: a ratio of 0.5 is also `<= 2`, but it has already returned 10 on the first line.

```python
def final_score(points_list):
    accuracy = sum(points_list) / (10 * len(points_list))
    return 60 * max(accuracy, 0)
```

Pass a **0 for every skipped photo**. Do not drop skipped photos from the list: they still count in the denominator, exactly as in the contest. Skipping everything gives score 0, not "undefined".

## 7. Mistakes to avoid

* Passing metres where pixels are expected (`focal_px`).
* Dropping skipped photos from `points_list` (inflates the score).
* Thinking "within 2/3 correct" is enough: it is enough only if all correct answers are in the +10 tier.
* Judging by the average error in metres. The score cares about tiers, and a few huge errors (−20) matter much more than many small ones.

## 8. Check yourself

1. At 200 m, what is σ? *(8.54 m)*
2. A photo at 350 m has an error of 25 m. How many points? *(ratio 25/9.57 = 2.6 → +3)*
3. You answer 10 photos: 7 earn +3 and 3 earn −20. Is it better than skipping all ten? *(21 − 60 = −39: no)*
4. Why can an excellent system still score 0? *(A few −20 answers cancel many +3 or +6 answers; see Section 4.)*
