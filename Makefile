# Common tasks. Everything runs through uv, so `make setup` is the only install step.
# Run `make` or `make help` for the list.
#
# The repository holds four Quarto books, which share code/, filters/, terms.toml and _common.yml:
#   book-full/          the full edition, with every derivation (website and PDF)
#   book-practitioner/  the practitioner edition, without proofs (website and PDF)
#   book-course/        the course: differential privacy one topic per chapter, every proof (website and PDF)
#   book-opendp/        the OpenDP book: the library from its first release to synthetic data (website and PDF)
# and the study notes, one page each, in notes/<name>/ (website and PDF). Each website renders to
# _book/<book>/ or _book/notes/<name>/, so the sites link to each other by relative path.

OUT       := _book
BUILD     := _build
BOOKS     := book-full book-practitioner book-course book-opendp
NOTE_DIRS := $(patsubst %/,%,$(wildcard notes/*/))
# The chapter scripts that draw figures. The practitioner workflows (practitioner_*.py) and the OpenDP book's
# scripts (opendp_*.py) are slow, need the datasets in data/ and draw no figures, so their tests run them instead
# (pytest marker `libs`). The study notes' scripts (code/notes/) run only when their note is built.
SCRIPTS   := $(foreach f,$(filter-out code/notes/%,$(wildcard code/*/*.py)),$(if $(or $(findstring /practitioner_,$(f)),$(findstring /opendp_,$(f))),,$(f)))
QUARTO    := uv run quarto
# The PDF file names, as each book's _quarto.yml sets them in `output-file`.
PDF_full         := $(OUT)/Privacy-Preserving-Machine-Learning.pdf
PDF_practitioner := $(OUT)/Privacy-Preserving-Machine-Learning-Practitioner.pdf
PDF_course       := $(OUT)/A-Course-in-Differential-Privacy.pdf
PDF_opendp       := $(OUT)/Differential-Privacy-with-OpenDP.pdf

# Keep Python's bytecode caches out of the source tree.
export PYTHONPYCACHEPREFIX := $(CURDIR)/$(BUILD)/pycache

# LaTeX packages the PDF needs beyond a minimal TinyTeX. The default TeX Live mirror on this
# machine is stale, so packages come from a fixed, current one.
TLMGR     := $(HOME)/.TinyTeX/bin/x86_64-linux/tlmgr
CTAN      := https://ctan.math.illinois.edu/systems/texlive/tlnet
TEX_PKGS  := svg imakeidx makeindex trimspaces catchfile transparent caption koma-script fontspec unicode-math \
             lualatex-math selnolig booktabs mdwtools footnotehyper xcolor framed fvextra upquote \
             microtype parskip xurl bookmark float tcolorbox pgf environ pdfcol tikzfill sidenotes \
             marginnote changepage pdflscape colortbl multirow wrapfig etoolbox fancyvrb amsmath amsfonts \
             tools geometry hyperref iftex luaotfload lm lm-math csquotes setspace \
             newcomputermodern dejavu

.DEFAULT_GOAL := help
.PHONY: help setup hooks tex-deps datasets test-fast figures figures-force full full-pdf practitioner practitioner-pdf course course-pdf opendp opendp-pdf \
        website pdf book notes note preview publish terms-check acronyms index-candidates prose ste prose-check duplicates duplicates-check notation notation-check \
        structure structure-check format test typecheck check all clean

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
define RUN_SCRIPT
	@mkdir -p $(@D)
	@echo "== $<"
	@uv run python $< >$@.tmp || { cat $@.tmp; rm -f $@.tmp; exit 1; }
	@mv $@.tmp $@
endef

$(BUILD)/logs/%.log: code/%.py $$(wildcard code/$$(dir $$*)*.py)
	$(RUN_SCRIPT)

# A study note's script stands alone, so it reruns only when it or the dataset loader changes, not
# when another note's script does (make picks this rule over the one above: its stem is shorter).
$(BUILD)/logs/notes/%.log: code/notes/%.py code/practitioner_datasets.py
	$(RUN_SCRIPT)

figures: $(FIG_LOGS) ## Generate the figures, rerunning only the chapter scripts that changed

figures-force: ## Rerun every chapter script, regenerating all the figures
	@rm -rf $(BUILD)/logs $(BUILD)/figures
	@$(MAKE) --no-print-directory figures

# A book is rendered from a staged copy in $(BUILD)/<name>-<format>/, never from its own folder,
# so Quarto's caches, LaTeX's intermediate files and the generated glossary and index stay out of
# the source tree. The staged copy sits two levels below the repository, as the rendered site does
# in $(OUT)/<book>/, so filters/links.lua can rewrite links that leave the book the same way for both.
# The shared settings (_common.yml, _book-common.yml) and filters/ are copied in next to the sources.
# STAGE(source folder, stage name)
define STAGE
	@mkdir -p $(BUILD)/$(2)
	@rsync -a --delete --exclude .quarto/ --exclude _book/ --exclude _site/ --exclude __pycache__/ $(1)/ $(BUILD)/$(2)/
	@rsync -a filters/ $(BUILD)/$(2)/filters/
	@cp _common.yml _book-common.yml $(BUILD)/$(2)/
	@for part in $(BUILD)/$(2)/part*/ $(BUILD)/$(2)/appendix*/; do \
	  [ -d "$$part" ] || continue; name=$$(basename "$$part"); \
	  if [ -d "$(BUILD)/figures/$$name" ]; then rsync -a $(BUILD)/figures/$$name/ $$part/figures/; fi; \
	done
endef

# RENDER(stage name, format): fails on any Quarto or LaTeX warning or error.
define RENDER
	@if ! $(QUARTO) render $(BUILD)/$(1) --to $(2) >$(BUILD)/$(1).log 2>&1; then \
	  cat $(BUILD)/$(1).log; exit 1; fi
	@if grep -E 'WARN|ERROR' $(BUILD)/$(1).log; then echo "$(1): warnings, see $(BUILD)/$(1).log"; exit 1; fi
endef

# BOOK_TARGETS(name, source folder): the targets `name` (website) and `name-pdf` for one book.
# tools/terms.py writes the book's glossary, index page and _terms.json into the staged copy first.
define BOOK_TARGETS
$(1): figures
	$$(call STAGE,$(2),$(1)-website)
	@uv run python tools/terms.py build --book $$(BUILD)/$(1)-website
	$$(call RENDER,$(1)-website,html)
	@mkdir -p $$(OUT)/$(2)
	@rsync -a --delete --exclude '*.pdf' $$(BUILD)/$(1)-website/_book/ $$(OUT)/$(2)/
	@echo "Website: $$(OUT)/$(2)/index.html"

$(1)-pdf: figures
	$$(call STAGE,$(2),$(1)-pdf)
	@uv run python tools/terms.py build --book $$(BUILD)/$(1)-pdf
	$$(call RENDER,$(1)-pdf,pdf)
	@mkdir -p $$(OUT)
	@cp $$(BUILD)/$(1)-pdf/_book/$$(notdir $$(PDF_$(1))) $$(PDF_$(1))
	@echo "PDF: $$(PDF_$(1))"
endef

$(eval $(call BOOK_TARGETS,full,book-full))
$(eval $(call BOOK_TARGETS,practitioner,book-practitioner))
$(eval $(call BOOK_TARGETS,course,book-course))
$(eval $(call BOOK_TARGETS,opendp,book-opendp))

full: ## Build the full edition's website into _book/book-full/
full-pdf: ## Build the full edition's PDF
practitioner: ## Build the practitioner edition's website into _book/book-practitioner/
practitioner-pdf: ## Build the practitioner edition's PDF
course: ## Build the course's website into _book/book-course/
course-pdf: ## Build the course's PDF
opendp: ## Build the OpenDP book's website into _book/book-opendp/
opendp-pdf: ## Build the OpenDP book's PDF

website: full practitioner course opendp ## Build every website into _book/

pdf: full-pdf practitioner-pdf course-pdf opendp-pdf ## Build every PDF into _book/

book: website pdf notes ## Build every website and PDF, and the study notes

# Each folder in notes/ is a one-page Quarto project (a standalone study note), staged and rendered
# like a book: its page and PDF go to _book/notes/<name>/. A note with computed content has a script,
# code/notes/<name with underscores>.py, which writes its figures and Markdown snippets into
# $(BUILD)/figures/notes/<name>/; it reruns only when it changes, and staging copies its output into
# the note's figures/, where the page includes it.
notes: ## Build the study notes in notes/ into _book/notes/
	@for dir in notes/*/; do name=$$(basename $$dir); \
	  $(MAKE) --no-print-directory note NOTE=$$name || exit 1; done

# A note whose page has more than one route can have a script per route: NOTE_ROUTES_<name> lists
# them, each route's script is code/notes/<name>_<route>.py, and its output is staged into figures/<route>/.
NOTE_SCRIPT = code/notes/$(subst -,_,$(NOTE)).py
NOTE_ROUTES_privacy-utility-tradeoff := synthetic

note:
	@for script in $(NOTE_SCRIPT) $(foreach r,$(NOTE_ROUTES_$(NOTE)),code/notes/$(subst -,_,$(NOTE))_$(r).py); do \
	  if [ -f $$script ]; then \
	    $(MAKE) --no-print-directory $$(echo $$script | sed 's|^code/\(.*\)\.py$$|$(BUILD)/logs/\1.log|') || exit 1; fi; done
	$(call STAGE,notes/$(NOTE),notes-$(NOTE))
	@if [ -d $(BUILD)/figures/notes/$(NOTE) ]; then \
	  rsync -a $(BUILD)/figures/notes/$(NOTE)/ $(BUILD)/notes-$(NOTE)/figures/; fi
	@for r in $(NOTE_ROUTES_$(NOTE)); do \
	  rsync -a --exclude runs/ $(BUILD)/figures/notes/$(NOTE)-$$r/ $(BUILD)/notes-$(NOTE)/figures/$$r/ || exit 1; done
	$(call RENDER,notes-$(NOTE),all)
	@mkdir -p $(OUT)/notes/$(NOTE)
	@rsync -a --delete $(BUILD)/notes-$(NOTE)/_site/ $(OUT)/notes/$(NOTE)/
	@echo "Note: $(OUT)/notes/$(NOTE)/index.html"

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
	$(QUARTO) preview $(BUILD)/full-website

# The checks below run over every book and every note unless BOOK= names one (a book folder, or
# notes/<name>); a page-level report takes PAGE= as well.
CHECKED   := $(or $(BOOK),$(BOOKS) $(NOTE_DIRS))

terms-check: ## Fail if an acronym isn't expanded at its first use on a page, or isn't in terms.toml (BOOK=)
	@for b in $(CHECKED); do echo "== $$b"; uv run python tools/terms.py check --book $$b || exit 1; done

acronyms: ## Report every acronym, its expansion and the pages that use it (BOOK=book-full)
	uv run python tools/terms.py acronyms --book $(or $(BOOK),book-full)

index-candidates: ## Suggest index terms (bold phrases) not yet in terms.toml (BOOK=book-full)
	uv run python tools/terms.py candidates --book $(or $(BOOK),book-full)

prose: ## Report hard-to-read prose, worst paragraphs first (BOOK=book-full, PAGE=part../NN-x.md); see PROSE.md
	uv run python tools/prose.py report --book $(or $(BOOK),book-full) $(if $(PAGE),--page $(PAGE)) --worst $(or $(WORST),5)

ste: ## Report departures from ASD-STE100 Simplified Technical English with stelint, never a gate (BOOK=, PAGE=); see PROSE.md rule 25
	uv run python tools/ste.py report --book $(or $(BOOK),book-practitioner) $(if $(PAGE),--page $(PAGE)) --worst $(or $(WORST),3) --rules

prose-check: ## Fail if a page's prose is past its book's limits in tools/prose.toml (BOOK=)
	@for b in $(CHECKED); do echo "== $$b"; uv run python tools/prose.py check --book $$b || exit 1; done

duplicates: ## Report paragraphs that repeat another paragraph, in every book and note (tools/duplicates.toml)
	uv run python tools/duplicates.py report

duplicates-check: ## Fail on a repeated paragraph that tools/duplicates.toml does not allow
	uv run python tools/duplicates.py check

notation-check: ## Fail on notation that NOTATION.md forbids, in every book and note (notation.toml)
	uv run python tools/notation.py check

notation: ## Report how often each notation rule fires, per book
	uv run python tools/notation.py report

structure-check: ## Fail if a page breaks its book's skeleton in tools/structure.toml (headings, ids, budget, pointer; BOOK=)
	uv run python tools/structure.py check $(if $(BOOK),--book $(BOOK))

structure: ## Report words per page and section, environments and budget, longest first (BOOK=book-full)
	uv run python tools/structure.py report --book $(or $(BOOK),book-full)

format: ## Format all Python with ruff, as VSCode and the pre-commit hook do
	uv run ruff check --fix --exit-zero --quiet .
	uv run ruff format .

test: datasets ## Run the tests, including the practitioner libraries' (marker `libs`)
	uv run pytest

test-fast: ## Run the tests without the practitioner libraries' (no torch, Opacus, OpenDP, ...)
	uv run pytest -m "not libs"

typecheck: ## Run strict pyright
	uv run pyright

check: test typecheck terms-check notation-check prose-check duplicates-check structure-check website ## Tests, types, acronyms, notation, prose, repeats, structure, and a clean render of every website

all: check pdf ## Check everything, then build the PDFs too

clean: ## Remove build output, staged copies and caches
	rm -rf $(OUT) $(BUILD)
