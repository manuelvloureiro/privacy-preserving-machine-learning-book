-- Cross-references into another book. The exercise book quotes the full edition's equations,
-- sections, figures and tables as @eq-name, @sec-name, ..., which Quarto can only resolve within
-- one book. tools/xrefs.py reads the rendered full edition and writes _xrefs.json into the staged
-- copy: for each label, its text ("Equation 20.3") and its page and anchor in the full edition.
-- This filter turns each such reference into a link with that text. A label the book defines
-- itself is left to Quarto.

local here = PANDOC_SCRIPT_FILE:match("(.*/)") or "./"
local file = io.open(here .. "../_xrefs.json", "r")
if not file then
  return {}
end
local data = quarto.json.decode(file:read("a"))
file:close()

-- The page's depth below the book's root, to climb from the page to the repository's level.
local input = quarto.doc.input_file
local project = quarto.project.directory or ""
local relative = input:sub(#project + 2)
local _, depth = relative:gsub("/", "")
local prefix = string.rep("../", depth + 1) .. data.book .. "/"

return {
  {
    Cite = function(cite)
      if #cite.citations ~= 1 then
        return nil
      end
      local ref = data.labels[cite.citations[1].id]
      if not ref then
        return nil
      end
      if quarto.doc.is_format("latex") then
        return pandoc.Str(ref.text)
      end
      return pandoc.Link(pandoc.Str(ref.text), prefix .. ref.href)
    end,
  },
}
