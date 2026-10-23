# Common tasks. Everything runs through uv, so `make setup` is the only install step.
# Run `make` or `make help` for the list.
#
# The repository holds four Quarto books, which share code/, filters/ and terms.toml:
#   book-full/          the full edition, with every derivation (website and PDF)
#   book-exercises/     the full edition's exercises with worked solutions (website only)
#   book-practitioner/  the practitioner edition, without proofs (website and PDF)
#   book-course/        the course: differential privacy one topic per chapter, every proof (website and PDF)
# Each website renders to _book/<book>/, so the sites link to each other by relative path.

OUT       := _book
BUILD     := _build
# The chapter scripts that draw figures. The practitioner workflows (practitioner_*.py) are slow, need the
# datasets in data/ and draw no figures, so their tests run them instead (pytest marker `libs`).
SCRIPTS   := $(foreach f,$(wildcard code/*/*.py),$(if $(findstring /practitioner_,$(f)),,$(f)))
QUARTO    := uv run quarto
PDF_full         := $(OUT)/Privacy-Preserving-Machine-Learning.pdf
PDF_practitioner := $(OUT)/Privacy-Preserving-Machine-Learning-Practitioner.pdf
PDF_course       := $(OUT)/A-Course-in-Differential-Privacy.pdf

# Keep Python's bytecode caches out of the source tree.
export PYTHONPYCACHEPREFIX := $(CURDIR)/$(BUILD)/pycache

# LaTeX packages the PDF needs beyond a minimal TinyTeX. The default TeX Live mirror on this
# machine is stale, so packages come from a fixed, current one.
TLMGR     := $(HOME)/.TinyTeX/bin/x86_64-linux/tlmgr
CTAN      := https://ctan.math.illinois.edu/systems/texlive/tlnet
TEX_PKGS  := svg imakeidx makeindex trimspaces catchfile transparent caption koma-script fontspec unicode-math \
             lualatex-math selnolig booktabs mdwtools footnotehyper xcolor framed fvextra upquote \
             microtype parskip xurl bookmark float tcolorbox pgf environ pdfcol tikzfill sidenotes \
             marginnote changepage colortbl multirow wrapfig etoolbox fancyvrb amsmath amsfonts \
             tools geometry hyperref iftex luaotfload lm lm-math csquotes setspace \
             newcomputermodern dejavu

.DEFAULT_GOAL := help
.PHONY: help setup hooks tex-deps datasets test-fast figures figures-force full full-pdf exercises practitioner practitioner-pdf course course-pdf \
        website pdf book preview publish terms-check acronyms index-candidates prose prose-check format test typecheck check all clean

help: ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-17s %s\n", $$1, $$2}'

setup: hooks ## Install the Python environment, Quarto included
	uv sync

hooks: ## Use the git hooks in .githooks/ (format staged files; reject commit messages that mention Claude)
	git config core.hooksPath .githooks

tex-deps: ## Install the LaTeX packages the PDF needs (also needs rsvg-convert: apt install librsvg2-bin)
	$(TLMGR) --repository $(CTAN) install $(TEX_PKGS)
	@command -v rsvg-convert >/dev/null || echo "rsvg-convert not found: sudo apt install librsvg2-bin"

datasets: ## Download the public datasets the practitioner scripts use into data/ (git-ignored), once
	uv run python code/practitioner_datasets.py fetch

# Figures are not committed: each script in code/<folder>/ writes its SVGs into
# $(BUILD)/figures/<folder>/, deterministically (seeded, no timestamps), and staging copies them into
# the book folder of the same name, <part>/figures/ or appendix-derivations/figures/. A script is
# rerun only when a file in its folder changes, since scripts import their siblings; its printed
# tables go to a log in $(BUILD)/logs/, which doubles as the up-to-date marker. Run with -j to regenerate in parallel.
FIG_LOGS  := $(patsubst code/%.py,$(BUILD)/logs/%.log,$(SCRIPTS))

.SECONDEXPANSION:
$(BUILD)/logs/%.log: code/%.py $$(wildcard code/$$(dir $$*)*.py)
	@mkdir -p $(@D)
	@echo "== $<"
	@uv run python $< >$@.tmp || { cat $@.tmp; rm -f $@.tmp; exit 1; }
	@mv $@.tmp $@

figures: $(FIG_LOGS) ## Generate the figures, rerunning only the chapter scripts that changed

figures-force: ## Rerun every chapter script, regenerating all the figures
	@rm -rf $(BUILD)/logs $(BUILD)/figures
	@$(MAKE) --no-print-directory figures

# A book is rendered from a staged copy in $(BUILD)/<name>-<profile>/, never from its own folder,
# so Quarto's caches, LaTeX's intermediate files and the generated glossary and index stay out of
# the source tree. The staged copy sits two levels below the repository, as the rendered site does
# in $(OUT)/<book>/, so filters/links.lua can rewrite links that leave the book the same way for both.
# STAGE(book folder, stage name, profile, extra rsync options)
define STAGE
	@mkdir -p $(BUILD)/$(2)
	@rsync -a --delete --exclude .quarto/ --exclude _book/ --exclude __pycache__/ $(4) $(1)/ $(BUILD)/$(2)/
	@rsync -a filters/ $(BUILD)/$(2)/filters/
	@for part in $(BUILD)/$(2)/part*/ $(BUILD)/$(2)/appendix*/; do \
	  [ -d "$$part" ] || continue; name=$$(basename "$$part"); \
	  if [ -d "$(BUILD)/figures/$$name" ]; then rsync -a $(BUILD)/figures/$$name/ $$part/figures/; fi; \
	done
endef

# RENDER(stage name, profile, format): fails on any Quarto or LaTeX warning or error.
define RENDER
	@if ! $(QUARTO) render $(BUILD)/$(1) --profile $(2) --to $(3) >$(BUILD)/$(1).log 2>&1; then \
	  cat $(BUILD)/$(1).log; exit 1; fi
	@if grep -E 'WARN|ERROR' $(BUILD)/$(1).log; then echo "$(1): warnings, see $(BUILD)/$(1).log"; exit 1; fi
endef

full: figures ## Build the full edition's website into _book/book-full/
	$(call STAGE,book-full,full-website,website)
	@uv run python tools/terms.py build --book $(BUILD)/full-website --profile website
	$(call RENDER,full-website,website,html)
	@mkdir -p $(OUT)/book-full
	@rsync -a --delete --exclude '*.pdf' $(BUILD)/full-website/_book/ $(OUT)/book-full/
	@echo "Website: $(OUT)/book-full/index.html"

full-pdf: figures ## Build the full edition's PDF
	$(call STAGE,book-full,full-pdf,pdf,--exclude _quarto-website.yml)
	@uv run python tools/terms.py build --book $(BUILD)/full-pdf --profile pdf
	$(call RENDER,full-pdf,pdf,pdf)
	@mkdir -p $(OUT)
	@cp $(BUILD)/full-pdf/_book/$(notdir $(PDF_full)) $(PDF_full)
	@echo "PDF: $(PDF_full)"

# The exercise book cross-references the full edition's equations, so it needs the rendered full
# edition: tools/xrefs.py reads its labels and numbers into _xrefs.json for filters/xrefs.lua.
exercises: full ## Build the exercise book's website into _book/book-exercises/ (builds the full edition first)
	$(call STAGE,book-exercises,exercises-website,website)
	@uv run python tools/xrefs.py --site $(OUT)/book-full --book book-full \
	  --skip-defined-in $(BUILD)/exercises-website --out $(BUILD)/exercises-website/_xrefs.json
	$(call RENDER,exercises-website,website,html)
	@mkdir -p $(OUT)/book-exercises
	@rsync -a --delete $(BUILD)/exercises-website/_book/ $(OUT)/book-exercises/
	@echo "Website: $(OUT)/book-exercises/index.html"

practitioner: figures ## Build the practitioner edition's website into _book/book-practitioner/
	$(call STAGE,book-practitioner,practitioner-website,website)
	@uv run python tools/terms.py build --book $(BUILD)/practitioner-website --profile website
	$(call RENDER,practitioner-website,website,html)
	@mkdir -p $(OUT)/book-practitioner
	@rsync -a --delete --exclude '*.pdf' $(BUILD)/practitioner-website/_book/ $(OUT)/book-practitioner/
	@echo "Website: $(OUT)/book-practitioner/index.html"

practitioner-pdf: figures ## Build the practitioner edition's PDF
	$(call STAGE,book-practitioner,practitioner-pdf,pdf,--exclude _quarto-website.yml)
	@uv run python tools/terms.py build --book $(BUILD)/practitioner-pdf --profile pdf
	$(call RENDER,practitioner-pdf,pdf,pdf)
	@mkdir -p $(OUT)
	@cp $(BUILD)/practitioner-pdf/_book/$(notdir $(PDF_practitioner)) $(PDF_practitioner)
	@echo "PDF: $(PDF_practitioner)"

course: figures ## Build the course's website into _book/book-course/
	$(call STAGE,book-course,course-website,website)
	@uv run python tools/terms.py build --book $(BUILD)/course-website --profile website
	$(call RENDER,course-website,website,html)
	@mkdir -p $(OUT)/book-course
	@rsync -a --delete --exclude '*.pdf' $(BUILD)/course-website/_book/ $(OUT)/book-course/
	@echo "Website: $(OUT)/book-course/index.html"

course-pdf: figures ## Build the course's PDF
	$(call STAGE,book-course,course-pdf,pdf,--exclude _quarto-website.yml)
	@uv run python tools/terms.py build --book $(BUILD)/course-pdf --profile pdf
	$(call RENDER,course-pdf,pdf,pdf)
	@mkdir -p $(OUT)
	@cp $(BUILD)/course-pdf/_book/$(notdir $(PDF_course)) $(PDF_course)
	@echo "PDF: $(PDF_course)"

website: full exercises $(if $(wildcard book-practitioner),practitioner) course ## Build every website into _book/

pdf: full-pdf $(if $(wildcard book-practitioner),practitioner-pdf) course-pdf ## Build every PDF into _book/

book: website pdf ## Build every website and PDF

# The site is the contents of $(OUT)/ plus site/, committed to the gh-pages branch from a worktree
# in $(BUILD)/gh-pages/ and served by GitHub Pages. site/index.html is the landing page, linking to
# the books and the PDFs; site/.nojekyll stops GitHub from
# running Jekyll over the site. It publishes whatever is in $(OUT)/, so build first.
PAGES     := $(BUILD)/gh-pages

publish: ## Publish _book/ to the gh-pages branch for GitHub Pages (run make book first)
	@test -d $(OUT)/book-full || { echo "Nothing to publish: run make book first"; exit 1; }
	@rm -rf $(PAGES) && git worktree prune
	@if git fetch -q origin gh-pages 2>/dev/null; then \
	  git worktree add -q -B gh-pages $(PAGES) FETCH_HEAD; \
	else \
	  git worktree add -q --orphan -b gh-pages $(PAGES); \
	fi
	@rsync -a --delete --exclude .git $(OUT)/ site/ $(PAGES)/
	@git -C $(PAGES) add -A
	@if git -C $(PAGES) diff --cached --quiet; then echo "The site is up to date"; else \
	  git -C $(PAGES) commit -q -m "Publish the books from $$(git rev-parse --short HEAD)" && \
	  git -C $(PAGES) push -q origin gh-pages && echo "Pushed to gh-pages"; fi
	@git worktree remove --force $(PAGES)

preview: full ## Serve the full edition from its staged copy (rerun after editing book-full/)
	$(QUARTO) preview $(BUILD)/full-website --profile website

terms-check: ## Fail if an acronym isn't expanded at its first use on a page, or isn't in terms.toml
	@for b in book-full book-exercises $(wildcard book-practitioner) book-course; do \
	  echo "== $$b"; uv run python tools/terms.py check --book $$b || exit 1; done

acronyms: ## Report every acronym, its expansion and the pages that use it (BOOK=book-full)
	uv run python tools/terms.py acronyms --book $(or $(BOOK),book-full)

index-candidates: ## Suggest index terms (bold phrases) not yet in terms.toml (BOOK=book-full)
	uv run python tools/terms.py candidates --book $(or $(BOOK),book-full)

prose: ## Report hard-to-read prose, worst paragraphs first (BOOK=book-practitioner, PAGE=part../NN-x.md); see PROSE.md
	uv run python tools/prose.py report --book $(or $(BOOK),book-practitioner) $(if $(PAGE),--page $(PAGE)) --worst $(or $(WORST),5)

prose-check: ## Fail if a page's prose is past its book's limits in tools/prose.toml (BOOK=book-practitioner)
	uv run python tools/prose.py check --book $(or $(BOOK),book-practitioner)

format: ## Format all Python with ruff, as VSCode and the pre-commit hook do
	uv run ruff check --fix --exit-zero --quiet .
	uv run ruff format .

test: datasets ## Run the tests, including the practitioner libraries' (marker `libs`)
	uv run pytest

test-fast: ## Run the tests without the practitioner libraries' (no torch, Opacus, OpenDP, ...)
	uv run pytest -m "not libs"

typecheck: ## Run strict pyright
	uv run pyright

check: test typecheck terms-check prose-check website ## Tests, type checks, the acronym and prose checks and a clean render of every website

all: check pdf ## Check everything, then build the PDFs too

clean: ## Remove build output, staged copies and caches
	rm -rf $(OUT) $(BUILD)
