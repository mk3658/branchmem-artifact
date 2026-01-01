# Annotation task: which value should an assistant's memory keep?

Thank you for helping. This should take about 45–60 minutes for 100 items.

Please **do not** read any other file in this folder or repository. Some of them
contain our own answers, and seeing them would make your work unusable.

## The situation

An AI assistant keeps a memory of facts about its user. The memory was copied
onto two devices — say a phone and a laptop — and both were used while offline,
so each copy was updated separately. Now they are back in sync and the two
copies disagree about the same fact. Something has to decide which version the
memory should keep.

Two things to know about the situation:

- **There are no timestamps.** The two devices had no shared clock while they
  were apart, so "which came later" is genuinely unavailable. Please do not try
  to infer it.
- **Each version is tagged with where it came from.** `source: user` means the
  user stated it directly. `source: inference` means the assistant guessed it
  from something else. `confidence` is the assistant's own confidence, from 0
  to 1. These tags are information you may use, or ignore, as you see fit.

You will see, for each item: the value the memory held before the two copies
diverged (`ancestor_value`), and the two competing values with their tags
(`branch_a_*` and `branch_b_*`).

## What to fill in

Open your assigned file — `discrimination_annotator1.csv`,
`...2.csv` or `...3.csv` — in Excel, Numbers, Google Sheets or any
spreadsheet program. Fill in the five answer columns for every row. Leave
everything else untouched.

**Q1 — `q1_which_value_should_be_stored`.** Type `a`, `b`, or
`neither_flag_for_review`.

Which value should the memory end up holding? Choose
`neither_flag_for_review` when you think nothing in front of you settles it and
a person should be asked — that is a real answer here, not a way of opting out.
Some items are built so that there is no determinate answer, and we want to
know which ones you think those are.

**Q2 — `q2_ancestor_is`.** Type `standing_rule`, `ordinary_value`, or `unclear`.

Think about the `ancestor_value` on its own. Is it the kind of entry that stays
in force until somebody explicitly withdraws it (`standing_rule`)? Or is it an
ordinary piece of information that a later, better update may simply replace
(`ordinary_value`)? Use `unclear` if it genuinely could be either.

**Q3 — `q3_who_could_lift_it`.** Only if you answered `standing_rule` to Q2.
Type `user_alone`, `someone_else`, `nobody_physical`, or `unclear`.

If that entry is a rule, who would have to agree before it stopped applying?

- `user_alone` — the user could simply decide to drop it; it is theirs.
- `someone_else` — somebody else would have to agree: a landlord, an employer,
  an insurer, a club, a venue, a library, and so on.
- `nobody_physical` — nobody can lift it, because it is a physical fact or a
  capacity limit rather than a decision.
- `unclear` — you cannot tell from what you are shown.

Leave Q3 blank when Q2 was not `standing_rule`.

**Q4 — `q4_confidence_1to5`.** A whole number from 1 to 5. 1 means you were
guessing; 5 means you were certain.

**`notes`** — optional, any length. Especially useful when you found an item
badly worded, ambiguous in a way the options do not capture, or when you
changed your mind.

## Please do

- Work through the items in the order they appear.
- Judge each item only on what that row shows you. Items are unrelated to each
  other and are deliberately in a mixed order.
- Answer from your own judgement. There is no key you are being measured
  against, and we are as interested in the items where people disagree as in
  the ones where they agree.
- Take breaks. Fatigue on a task like this is a real source of noise.

## Please do not

- Do not look anything up, and do not ask a chatbot. We need human judgement
  specifically; model judgements we already have.
- Do not discuss the items with the other annotators until all three of you have
  finished.
- Do not leave answer cells empty for Q1, Q2 or Q4.

## When you are done

Save the file in CSV format, keeping the same filename, and send it back.
Please also tell us roughly how long it took you.

If anything about the task is unclear, ask before starting rather than
guessing at what we wanted — a question now is cheaper than 100 rows
annotated under a misunderstanding.
