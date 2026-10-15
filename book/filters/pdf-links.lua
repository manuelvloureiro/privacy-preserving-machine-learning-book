-- PDF only: drop links that have no target in the PDF.
--
-- * The exercise book (the solutions/ pages, with every chapter's exercises and their worked
--   solutions) is published on the website only (see _quarto-pdf.yml). A link to it keeps its text,
--   followed by "in the web edition": "Exercise 3 shows ..." becomes "Exercise 3 in the web edition
--   shows ...".
-- * A part page (the 00-introduction.md of a partNN-* folder) becomes \part{...}, which Quarto gives no label,
--   so a link to it keeps its text and loses the link.

if not quarto.doc.is_format("latex") then
  return {}
end

return {
  {
    Link = function(link)
      if link.target:match("solutions/") then
        local content = link.content:clone()
        content:extend({ pandoc.Space(), pandoc.Str("in"), pandoc.Space(), pandoc.Str("the"), pandoc.Space(),
          pandoc.Str("web"), pandoc.Space(), pandoc.Str("edition") })
        return content
      end
      if link.target:match("00%-introduction%.md$") then
        return link.content
      end
    end,
  },
}
