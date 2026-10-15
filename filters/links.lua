-- Links that leave this book. The repository holds three books side by side (book-full/,
-- book-exercises/, book-practitioner/) next to code/ and tests/, and a page links to them by
-- relative path from its source file, so the links also work when browsing the repository.
--
-- * The websites render to _book/<book>/, mirroring the source tree one level down, and the
--   staged copies sit in _build/<book>-<profile>/. A link out to code/ or tests/ therefore gains
--   one "../", and a link to another book only swaps ".md" for ".html".
-- * PDF only: a link to another book keeps its text, followed by where to find it: "Exercise 3
--   shows ..." becomes "Exercise 3 in the exercise book shows ...". A part page (the
--   00-introduction.md of a partNN-* folder) becomes \part{...}, which Quarto gives no label,
--   so a link to it keeps its text and loses the link.

local is_pdf = quarto.doc.is_format("latex")

local where = {
  ["book-full"] = "in the full edition",
  ["book-exercises"] = "in the exercise book",
  ["book-practitioner"] = "in the practitioner edition",
}

return {
  {
    Link = function(link)
      -- Lua patterns can't repeat a group, so strip the leading "../"s one at a time.
      local ups, rest = "", link.target
      while rest:sub(1, 3) == "../" do
        ups, rest = ups .. "../", rest:sub(4)
      end
      if ups ~= "" then
        local top = rest:match("^([^/]+)/")
        if top == "code" or top == "tests" then
          link.target = "../" .. link.target
          return link
        end
        local book, path, anchor = rest:match("^(book%-[%a]+)/(.-)%.md(#?.*)$")
        if book and where[book] then
          if is_pdf then
            local content = link.content:clone()
            content:insert(pandoc.Space())
            content:extend(pandoc.Inlines(where[book]))
            return content
          end
          link.target = ups .. book .. "/" .. path .. ".html" .. anchor
          return link
        end
      end
      if is_pdf and link.target:match("00%-introduction%.md$") then
        return link.content
      end
    end,
  },
}
