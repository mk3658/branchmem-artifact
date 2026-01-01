# Instructions for independent annotators

You are deciding what an assistant should store after two offline copies of its
memory disagreed. There is no shared clock, so timestamps are unavailable.
Read the stored ancestor entry, both candidate branch values, and their source
and confidence metadata. Work independently; do not discuss answers with other
annotators or use an LLM to answer. No particular answer is expected.

Use your assigned CSV and keep its rows, IDs, inputs, and column names intact.
Fill these answer columns:

- q1_which_value_should_be_stored: a, b, or neither_flag_for_review.
  Choose the value to store, or flag the conflict if neither can be selected.
- q2_ancestor_is: standing_rule, ordinary_value, or unclear.
  Is the stored ancestor a rule that remains in force until withdrawn, an
  ordinary value that may be replaced, or unclear?
- q3_who_could_lift_it: user_alone, someone_else, nobody_physical, or unclear.
  Answer if Q2 is standing_rule; otherwise leave this cell blank.
- q4_confidence_1to5: integer 1 (guess) to 5 (certain).
- notes: optional explanation of uncertainty or concerns.

Complete all 220 items. You may pause and resume. Do not guess simply to avoid
an unclear or review answer. Return the completed CSV to the study organizer.
The organizer should obtain informed consent, pay at least the applicable local
minimum wage for time spent, and identify annotators using random IDs rather
than names. No demographic information is needed for these sheets.
