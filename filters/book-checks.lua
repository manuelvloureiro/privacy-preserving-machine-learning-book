-- Runs before Quarto's own filters (see _quarto.yml).
--
-- 1. Numbers every display equation: one without an {#eq-...} label gets an automatic one,
--    so Quarto numbers it like a labelled one. Label an equation yourself only to reference it.
-- 2. Warns about any table or figure without a caption, so a missing caption shows up in the build.

local doc_name = quarto.doc.input_file:match("[^/]+$")
local counter = 0

local function is_label(inline)
  return inline and inline.t == "Str" and inline.text:match("^{#eq%-")
end

local function number_equations(inlines)
  local out = pandoc.Inlines({})
  local i = 1
  while i <= #inlines do
    local el = inlines[i]
    out:insert(el)
    if el.t == "Math" and el.mathtype == "DisplayMath" then
      local nxt, after = inlines[i + 1], inlines[i + 2]
      local labelled = is_label(nxt) or (nxt and nxt.t == "Space" and is_label(after))
      if not labelled then
        counter = counter + 1
        local stem = doc_name:gsub("%.q?md$", ""):gsub("[^%w]", "")
        out:insert(pandoc.Space())
        out:insert(pandoc.Str("{#eq-auto-" .. stem .. "-" .. counter .. "}"))
      end
    end
    i = i + 1
  end
  return out
end

local function stringify_caption(caption)
  if caption == nil then return "" end
  return pandoc.utils.stringify(caption.long or caption)
end

local function warn(message)
  io.stderr:write("WARNING: " .. doc_name .. ": " .. message .. "\n")
end

-- Quarto wraps a labelled table or figure in a FloatRefTarget and moves its caption there,
-- so the inner Table or Figure looks uncaptioned. Check the wrapper and don't descend into it.
local caption_checks = {
  traverse = "topdown",
  FloatRefTarget = function(float)
    if stringify_caption(float.caption_long) == "" then
      warn(string.lower(float.type) .. " " .. (float.identifier or "") .. " without a caption")
    end
    return float, false
  end,
  Table = function(tbl)
    if stringify_caption(tbl.caption) == "" then
      warn("table without a caption (header: " .. pandoc.utils.stringify(tbl.head):sub(1, 60) .. ")")
    end
  end,
  Figure = function(fig)
    if stringify_caption(fig.caption) == "" then
      warn("figure without a caption")
    end
  end,
}

return {
  { Inlines = number_equations },
  caption_checks,
}
