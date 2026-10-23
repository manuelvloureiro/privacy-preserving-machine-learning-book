# Cold-reader test

Use this prompt with a reviewer (a person, or a model in a fresh context) that has **not** seen the
full edition, the other pages, or the writer's notes. Give it only the page and the audience line.
The test finds the places where a reader of that audience gets lost. It is stage 3 of PROSE.md.

---

You are a careful reader of a technical book. Audience: {AUDIENCE}. You know what the earlier
chapters' titles promise ({EARLIER_TITLES}), but you have not read them, and you have nothing
else: no internet, no other pages.

Read the page below once, from the top, the way that reader would. For **every paragraph**, in order:

1. Restate what the paragraph says in one plain sentence of your own. If you cannot, say so.
2. List every **stall**: a place where you had to stop, reread, or guess. For each, quote the exact
   words (at most 20) and say which kind it is:
   - **undefined**: a term, symbol or acronym used before the page says what it means;
   - **jump**: a step from one claim to the next that the text does not explain;
   - **overload**: too many new things at once to hold in mind;
   - **ambiguous**: a pronoun, "this", or a phrase whose referent is unclear;
   - **unmotivated**: a result or formula that appears without saying what question it answers;
   - **garbled**: grammar or word order that makes the sentence hard to parse;
   - **unstated assumption**: a condition a result needs (non-negative, independent, bounded, fixed in
     advance) stated as if it were a fact about the world, or without saying why the result needs it;
   - **purposeless section**: a section or heading that does not say what question it answers or why
     you need the answer now, so you cannot tell what to look for in it;
   - **out of order**: a result used before it is proved, a theorem whose proof sits in another section,
     or a lemma whose use you cannot yet see;
   - **packed display**: a displayed formula that makes several moves at once (simplifies, then bounds,
     then substitutes), or whose reason is given only after it;
   - **formula without words**: a passage you could not follow if the formulas were replaced by "blah".
3. Say whether the paragraph follows from the one before it (yes / weakly / no).

Do not judge whether the content is correct, and do not suggest rewrites. Be literal: if you
understood something only because you already knew it, count it as a stall for the stated audience.

End with the three paragraphs that were hardest to read, and why. Then, under **Rhythm**, quote any
run of three or more sentences that reads choppy or stilted: short sentences in a row that should be
one, or the same opening repeated.

Page:

{PAGE}
