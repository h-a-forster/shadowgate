# Review of shared errors (MMLU-Pro, Haiku -> Opus)

A shared error is a skipped case (Haiku confidence >= 0.80) where Haiku's answer is wrong against
the MMLU-Pro gold letter and Opus gave the same letter. The audit cannot see these. There are 55.
Thirteen were drawn at random (`random.seed(7)`, 6 cases, then `random.seed(8)`, 7 more from the
rest) and read by hand. The verdicts are one reviewer's judgement, not a second annotation pass.

| Question id | Subject | Gold | Both models | Verdict |
|---|---|---|---|---|
| 12010 | engineering | C | H | gold wrong: 100/(7.07+j7.07) = 7.07-j7.07 ohm, option H |
| 7223 | economics | A | J | gold wrong: an imported good lowers net exports and leaves GDP unchanged (J) |
| 9951 | physics | C (1.0 m) | F (9.0 m) | gold wrong: 90 x 6 = 60 x d gives 9.0 m |
| 2700 | psychology | E (episodic) | A (implicit) | gold wrong: implicit memory is the standard "most automatic" answer |
| 2129 | psychology | A (hemiplegia) | B (anxiety disorders) | gold wrong: hemiplegia is not a behaviour disorder |
| 10252 | physics | F (more than doubles) | E (quadruples) | ambiguous: both options are true |
| 11036 | philosophy | B | C | ambiguous: B and C are logically equivalent |
| 2173 | psychology | D | A | ambiguous: A and D state the same result |
| 7001 | economics | C (Gosbank) | H (central planning) | ambiguous: H is at least as defensible |
| 6339 | health | J (< 50%) | H (< 25%) | ambiguous: depends on the source the item assumes |
| 6277 | health | G (pigs) | A (wild birds) | ambiguous: wild birds are the reservoir; pigs the mixing vessel |
| 6937 | economics | E | D | likely a real model error: E is the textbook answer |
| 1370 | law | G | B | likely a real model error (gold plausible) |

Summary: 5 of 13 gold labels are wrong, 6 are ambiguous or have more than one defensible option,
and 2 look like real errors by both models. This shows that label noise accounts for some of the
gap between Haiku's gold-graded error (9.3%) and its disagreement with Opus (6.1%), not how much.
Thirteen cases from two ad hoc seeds and one reviewer are too few to size it, and the check only
read cases graded wrong: a wrong gold label can also mark a wrong Haiku answer as right, which
would push true error up. So the gold-graded rate is not shown to be an upper bound, and Haiku's
true error on skipped cases could lie below the disagreement rate or above the gold-graded rate.
See [results](../../docs/results.md) for how this feeds into the headline numbers.
