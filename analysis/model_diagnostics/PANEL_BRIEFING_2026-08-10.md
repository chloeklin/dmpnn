# Panel chair meeting — briefing notes

**Monday 10 August 2026.** Audience: panel chair, not primary supervisor. He is checking that
the work is sound and on track, and that I can explain it. He cannot and should not be asked to
approve the dataset decision — that goes to my primary supervisor.

**Pitch level:** intelligent scientist, not in this subfield. Plain language. No metric names
without a one-line explanation.

**What I want him to leave thinking:** the project has a clear result, I found its weaknesses
myself before anyone else did, and I know what the next twelve months look like.

---

## Opening — 60 seconds

> "Short version: I have a model that beats the published state of the art on the standard
> copolymer benchmark, and the improvement is statistically significant, which is rare in this
> area. But I've found a reason to be cautious about *why* it wins, and most of my work this
> month has been testing that rather than chasing more accuracy."

That framing does the work. It says: I have a result, and I am not overselling it.

---

## Topic 1 — How the benchmark dataset was made, and what I could build

### The problem, in one paragraph

Polymers aren't single molecules. A copolymer made of monomers A and B is really a whole
population of chains that differ in how A and B are ordered. Two materials can have identical
chemistry and identical composition but different *architecture* — blocky, alternating, or
random — and different properties. The question my work asks is whether machine learning models
can actually represent that difference.

### How the existing labels were computed

The benchmark I use has 42,966 copolymers with two computed electronic properties. The authors
generated it like this:

1. **Build the chains.** Join the two monomers into an **8-unit chain**, respecting the
   requested composition and arrangement. Because many orderings satisfy the same recipe,
   generate **up to 32 different chains** per polymer.
2. **Make 3D shapes.** Each chain is floppy, so generate **8 conformations** per chain — up to
   256 structures per polymer.
3. **Optimise the geometry** — rough force-field pass, then quantum chemistry.
4. **Compute the properties** for each structure.
5. **Average** — over the 8 shapes, then over all the chains. One number per polymer.

Steps 3 and 4 are the expensive part. Everything before is fast.

### Could I build something like it?

Yes. The authors published their generation code and I've read it. It runs on the monomers we
already have, because it encodes a real chemical reaction — Suzuki coupling — and our monomers
are exactly the right type for it.

| | |
|---|---|
| worst case per polymer | 256 structures |
| ~2,000 polymers | ≤ 512,000 structures |
| at ~30 CPU-seconds each | **≈ 8–9 kSU of CPU** |

That is a **ceiling** — the real figure is lower, and a one-day pilot would measure it. For
scale, reproducing the full 42,966 would be ~150 kSU. I'd be proposing about 5%.

**Important: this is CPU, not GPU.** Different queue from my model training.

### What I'd actually build, and why

Not a bigger version of the same thing. The gap I can demonstrate is this: **the existing
benchmark has only three architecture settings.** I checked — there are literally three distinct
arrangement descriptions across all 42,966 rows. So a model only ever has to tell two or three
cases apart.

I'd generate a version where **blockiness varies continuously** at fixed chemistry and fixed
composition — polymers that are 20% blocky, 40% blocky, and so on. That turns a three-way
classification into a real measurement axis.

I'd also publish the **un-averaged values** — one number per chain rather than only the average.
That costs nothing extra, since the calculation already produces them, and it's strictly more
information than any existing benchmark provides.

**If asked whether this is a good use of time:** it's a scoping question for my primary
supervisor. I have it costed and designed; I'm not asking for a decision today.

---

## Topic 2 — Why does my model beat the baseline?

### The result first

On the harder chemistry-extrapolation split, against the baseline running at **its own authors'
published settings**:

| | baseline | my model |
|---|---|---|
| prediction error (EA) | 0.070 eV | **0.055 eV** — 21% lower |
| prediction error (IP) | 0.050 eV | **0.035 eV** — 30% lower |
| architecture recovery (EA) | 0.397 | **0.849** |
| architecture recovery (IP) | 0.565 | **0.886** |

Two of these hold on **9 out of 9 folds**, which is significant at p = 0.004 — the best
achievable with nine folds. Significance is rare in this literature and I want to state it
carefully rather than lean on it.

**One thing I did that I think matters:** the obvious objection is "your baseline was badly
configured." So I went back to the original paper, found the exact settings they used, and
re-ran it that way. **The baseline got better** — noticeably better at the chemistry part. And
my model still won on architecture. That objection is now closed.

### Why does it win? — honest answer: I don't fully know yet

My model differs from the simpler version in five ways at once. I've been eliminating them one
at a time.

| factor | status |
|---|---|
| averaging over 16 sampled chains | **ruled out** — tested, made no difference |
| learned position information | **testing now**, results this week |
| how chain information is pooled | untested |
| the 8-unit chain structure itself | untested |
| it *discards* junction chemistry the simpler model uses | untested |

**The ruled-out one is worth explaining**, because it's the cleanest piece of work this month.
I pre-registered the experiment — wrote down in advance what each possible outcome would mean,
so I couldn't rationalise afterwards. The prediction was that the averaging was doing the work.
It wasn't. That's a negative result, and I'm reporting it as one.

**A limit I should be upfront about:** the remaining differences are small enough that they sit
inside the run-to-run noise of this dataset. So there's a point beyond which no further ablation
on this data can separate them. That's a property of the benchmark, not of effort.

---

## Topic 3 — The confound I found in my own work

**This is the part I most want to raise myself.**

The benchmark's labels were computed on **8-unit chains**, averaged over **up to 32 sampled
arrangements**. My best model uses an **8-slot chain**, samples **16 arrangements**, and
**averages** its predictions.

**My model's structure mirrors how the labels were made.**

Two readings, both legitimate:

- **Favourable** — I encoded the right physics. The property genuinely *is* an average over
  chain arrangements, so a model built that way should win. Matching a model to the process
  that generated the data is good science.
- **Sceptical** — the advantage may not be "this is a better polymer representation," but "this
  happens to match this dataset's recipe." Real polymers are hundreds of units long; real
  measured properties are not 8-unit averages.

**I cannot distinguish these on this dataset.** No ablation resolves it. It's the single biggest
threat to the claim, and I'd rather name it than have a reviewer find it.

### What I can do about it — four options, ranked

**1. Test on a different dataset.** I already have two — a glass-transition dataset and a
block-copolymer phase dataset — whose labels have nothing to do with 8-unit averaging. If my
model still wins there, the sceptical reading weakens a lot. **Cheapest real test; days of GPU,
no new data.**

**2. Vary the chain length in my model** — 8 versus 12 versus 16 slots. If performance peaks
sharply at exactly 8, that's protocol-matching. If it's flat or improves with more, it isn't.
One caution I've already checked: the compositions are quarters, so 6 and 10 slots can't
represent them exactly and would look worse for purely arithmetic reasons. **8, 12, 16 and 24
are the clean comparisons.**

**3. Ask what the representation contains, not just what it predicts.** Freeze the model,
then test whether architecture can be read out of its internal representation by a simple linear
classifier. That separates "predicts better" from "represents better," which is the actual
claim. **One afternoon.**

**4. Generate labels differently** — the dataset proposal above. Definitive, but months and
~9 kSU, and it needs sign-off.

**My inclination is 1 and 3 first** — both are cheap, both use things I already have, and
together they'd move the claim a long way.

---

## Am I on track?

**Delivered this year:** a review paper accepted at *Digital Discovery*; a two-axis evaluation
framework with a null-model floor; ~590 training runs under a frozen, pre-registered protocol; a
measured run-to-run noise floor, which nobody in this literature publishes; and a model that
significantly beats the published baseline at that baseline's own settings.

**Found and fixed this year:** a model-selection bug affecting every experiment, an input-parsing
bug that silently disabled the baseline's polymer features, a mis-specified quality filter, and a
performance defect that made all timing comparisons wrong. All documented, all corrected, all
disclosed.

**Next twelve months:** submit the measurement paper; run the cheap external-validity tests
above; decide with my primary supervisor whether the dataset is worth building.

**Milestone document is due 29 August.** On track — I'm writing to a fixed outline and not
running new experiments for it.

---

## Anticipated questions

**"Is a 20% error reduction meaningful?"**
On its own, modest. What matters more is that on the hardest split my model and the baseline are
*equally* accurate but differ enormously on architecture recovery — same predictions, two
measures pointing opposite ways. That's the argument: the number everyone reports can't see what
I'm studying.

**"Why not just use more data?"**
Tested — accuracy barely depends on training-set size here. The limit isn't data volume, it's
that architecture is about 1% of total variation. That's exactly why I built a metric that looks
at the residual instead.

**"Have you shown this to the original authors?"**
Not yet. Worth doing on one point — their published code doesn't record which version of the
quantum chemistry program produced the labels, which matters if anyone extends the dataset.

**"What if the confound turns out to be real?"**
Then it's a finding, not a failure — that a model's advantage can come from matching a dataset's
generation protocol is worth reporting, and it would be a caution for the whole subfield. Either
way I have a result. That's why I'm testing it rather than avoiding it.

**"What do you need?"**
Nothing today. One decision is pending with my primary supervisor — whether to build the dataset
— and I have it costed and designed for when we discuss it.

---

## Do not do

- Don't lead with metric names. Say "architecture recovery," not ΔR², unless asked.
- Don't bury the confound at the end. It's the strongest evidence of scientific judgement here.
- Don't ask him to decide the dataset question. Wrong person, and it makes it look like I'm
  waiting for permission rather than working.
- Don't present the position-embedding result as though it's in. It isn't yet.
