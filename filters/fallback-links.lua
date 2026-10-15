-- The practitioner edition only: a link to one of its own chapters that isn't written yet goes to
-- the same chapter of the full edition instead, so pages can link to their final targets while the
-- edition is being written. On a website the link points to the full edition's page; in a PDF it
-- becomes plain text followed by "in the full edition". Once the chapter exists, the link is left
-- alone. Runs before filters/links.lua and Quarto's own filters.

local project = quarto.project.directory or ""
local input = quarto.doc.input_file
local page_dir = input:match("(.*)/") or "."

-- The page's depth below the book's root, to climb from the page to the repository's level.
local relative_page = input:sub(#project + 2)
local _, depth = relative_page:gsub("/", "")

-- Resolve "a/b/../c.md" against the book's root, giving "a/c.md", or nil if it leaves the book.
local function book_path(target)
  local parts = {}
  for part in (page_dir:sub(#project + 2) .. "/" .. target):gmatch("[^/]+") do
    if part == ".." then
      if #parts == 0 then
        return nil
      end
      table.remove(parts)
    elseif part ~= "." then
      table.insert(parts, part)
    end
  end
  return table.concat(parts, "/")
end

local function exists(path)
  local file = io.open(path, "r")
  if file then
    file:close()
    return true
  end
  return false
end

return {
  {
    Link = function(link)
      local target, anchor = link.target:match("^([^#:]+%.md)(#?.*)$")
      if not target then
        return nil
      end
      local path = book_path(target)
      if not path or exists(project .. "/" .. path) then
        return nil
      end
      if quarto.doc.is_format("latex") then
        local content = link.content:clone()
        content:insert(pandoc.Space())
        content:extend(pandoc.Inlines("in the full edition"))
        return content
      end
      link.target = string.rep("../", depth + 1) .. "book-full/" .. path:gsub("%.md$", ".html") .. anchor
      return link
    end,
  },
}
