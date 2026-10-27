# Cold-reader test

Use this prompt with a reviewer (a person, or a model in a fresh context) that has **not** seen the
full edition, the other pages, or the writer's notes. Give it only the page and the audience line.
The test finds the places where a reader of that audience gets lost, and above all every connection
the page leaves the reader to infer. It is stage 3 of PROSE.md. `uv run python tools/prose.py prompt
--book BOOK --page PAGE` prints this prompt filled in for one page.

---

You are a careful reader of a technical book. Audience: {AUDIENCE}. You know what the earlier
chapters' titles promise ({EARLIER_TITLES}), but you have not read them, and you have nothing
else: no internet, no other pages.

Read the page below once, from the top, the way that reader would. For **every paragraph**, in order:

1. Restate what the paragraph says in one plain sentence of your own. If you cannot, say so. Then
   say whether the paragraph's first sentence already told you that point (yes / no).
2. **Inference audit.** For each sentence after the first, write what you had to supply yourself to
   get from the sentence before to this one: a fact you had to recall from earlier, a step of
   algebra or arithmetic you had to do, a reason the text did not give, a referent you had to guess
   ("the straight line": which one?), or a conclusion stated before its argument. Each of these is an
   **inference call**. Write "none" when the sentence follows with nothing left to you. Count the
   inference calls of the paragraph; this count is the test's main result.
3. List every **stall**: a place where you had to stop, reread, or guess. For each, quote the exact
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
4. Say whether the paragraph follows from the one before it (yes / weakly / no), and name any
   sentence that is an aside, a detail that does not serve the paragraph's point.

Do not judge whether the content is correct, and do not suggest rewrites. Be literal: if you
understood something only because you already knew it, count it as a stall for the stated audience.

End with a table of paragraphs and their inference-call counts, the total for the page, and the
three paragraphs that were hardest to read, and why. Then, under **Rhythm**, quote any
run of three or more sentences that reads choppy or stilted: short sentences in a row that should be
one, or the same opening repeated.

Page:

{PAGE}
