-- PDF only: put \index{term} at the start of every paragraph that mentions an index term,
-- so LaTeX's index carries real page numbers. The terms come from _terms.json, which
-- tools/terms.py writes from terms.toml. The web index is the generated term-index.md.

if not quarto.doc.is_format("latex") then
  return {}
end

local here = PANDOC_SCRIPT_FILE:match("(.*/)") or "./"
local file = io.open(here .. "../_terms.json", "r")
if not file then
  return {}
end
local data = quarto.json.decode(file:read("a"))
file:close()

-- The PDF book is one document: each page starts at a level-1 heading. Skip the pages
-- whose first heading id is listed (the study plan, the glossary, the index itself).
local skip = {}
for _, id in ipairs(data.skip or {}) do
  skip[id] = true
end

-- Lua patterns: escape magic characters, let a space match a space or a hyphen, allow plurals.
local function compile(variant)
  local escaped = variant:gsub("[%^%$%(%)%%%.%[%]%*%+%-%?]", "%%%0"):gsub(" ", "[%%s%%-]+")
  return "%f[%w]" .. escaped .. "e?s?%f[^%w]"
end

local terms = {}
for term, variants in pairs(data.terms) do
  local patterns = {}
  for _, v in ipairs(variants) do
    table.insert(patterns, compile(v))
  end
  -- "sortkey@Shown": sort case-insensitively, show the term as written. Quote makeindex's specials.
  local shown = term:gsub('(["!@|])', '"%1')
  table.insert(terms, { entry = shown:lower() .. "@" .. shown, patterns = patterns })
end

local function mark(block)
  local text = pandoc.utils.stringify(block):lower()
  local marks = {}
  for _, t in ipairs(terms) do
    for _, p in ipairs(t.patterns) do
      if text:find(p) then
        table.insert(marks, pandoc.RawInline("latex", "\\index{" .. t.entry .. "}"))
        break
      end
    end
  end
  if #marks > 0 then
    block.content = pandoc.List(marks) .. block.content
    return block
  end
end

return { {
  Pandoc = function(doc)
    local skipping = false
    for i, block in ipairs(doc.blocks) do
      if block.t == "Header" and block.level == 1 then
        skipping = skip[block.identifier] or false
      elseif not skipping then
        doc.blocks[i] = block:walk({ Para = mark, Plain = mark })
      end
    end
    return doc
  end,
} }
