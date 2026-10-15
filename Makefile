# Common tasks. Everything runs through uv, so `make setup` is the only install step.
# Run `make` or `make help` for the list.

BOOK      := book
OUT       := _book
PDF       := $(OUT)/Privacy-Preserving-Machine-Learning.pdf
SCRIPTS   := $(wildcard $(BOOK)/*/code/*.py)
QUARTO    := uv run quarto
BUILD     := _build
STAGE_WEB := $(BUILD)/website
STAGE_PDF := $(BUILD)/pdf

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
.PHONY: help setup hooks tex-deps figures figures-force stage-tests stage-website stage-pdf terms terms-check acronyms index-candidates book website pdf preview check-captions format test typecheck check all clean

help: ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-17s %s\n", $$1, $$2}'

setup: hooks ## Install the Python environment, Quarto included
	uv sync

hooks: ## Use the git hooks in .githooks/ (format staged files; reject commit messages that mention Claude)
	git config core.hooksPath .githooks

tex-deps: ## Install the LaTeX packages the PDF needs (also needs rsvg-convert: apt install librsvg2-bin)
	$(TLMGR) --repository $(CTAN) install $(TEX_PKGS)
	@command -v rsvg-convert >/dev/null || echo "rsvg-convert not found: sudo apt install librsvg2-bin"

# Figures are not committed: each chapter script writes its SVGs into its part's figures/ folder,
# deterministically (seeded, no timestamps). A script is rerun only when a file in its code/ folder
# changes, since scripts import their siblings; its printed tables go to a log in $(BUILD)/figures/,
# which doubles as the up-to-date marker. Run with -j to regenerate in parallel.
FIG_LOGS  := $(patsubst $(BOOK)/%.py,$(BUILD)/figures/%.log,$(SCRIPTS))

.SECONDEXPANSION:
$(BUILD)/figures/%.log: $(BOOK)/%.py $$(wildcard $(BOOK)/$$(dir $$*)*.py)
	@mkdir -p $(@D)
	@echo "== $<"
	@uv run python $< >$@.tmp || { cat $@.tmp; rm -f $@.tmp; exit 1; }
	@mv $@.tmp $@

figures: $(FIG_LOGS) ## Generate the figures, rerunning only the chapter scripts that changed

figures-force: ## Rerun every chapter script, regenerating all the figures
	@rm -rf $(BUILD)/figures
	@$(MAKE) --no-print-directory figures

terms: stage-website stage-pdf ## Regenerate the glossary and the index of terms in the staged copies

terms-check: ## Fail if an acronym isn't expanded at its first use on a page, or isn't in terms.toml
	uv run python tools/terms.py check

acronyms: ## Report every acronym, its expansion and the pages that use it
	uv run python tools/terms.py acronyms

index-candidates: ## Suggest index terms (bold phrases) not yet in terms.toml
	uv run python tools/terms.py candidates

# The book is rendered from staged copies of book/ in _build/<profile>/, never from book/ itself,
# so Quarto's caches, LaTeX's intermediate files and the generated glossary and index stay out of
# the source tree. The rendered website and PDF are then copied to _book/. A render fails on any
# Quarto or LaTeX warning or error.
RENDER_CHECK = if ! $(QUARTO) render $(1) --profile $(2) --to $(3) >$(BUILD)/$(2).log 2>&1; then \
	  cat $(BUILD)/$(2).log; exit 1; fi; \
	if grep -E 'WARN|ERROR' $(BUILD)/$(2).log; then echo "$(2): warnings, see $(BUILD)/$(2).log"; exit 1; fi

# Chapters link to their tests as ../../tests/...; from a staged copy that path is _build/tests.
stage-tests:
	@mkdir -p $(BUILD)
	@ln -sfn ../tests $(BUILD)/tests

stage-website: stage-tests figures
	@mkdir -p $(STAGE_WEB)
	@rsync -a --delete --exclude .quarto/ --exclude _book/ --exclude __pycache__/ $(BOOK)/ $(STAGE_WEB)/
	@uv run python tools/terms.py build --book $(STAGE_WEB) --profile website

stage-pdf: stage-tests figures
	@mkdir -p $(STAGE_PDF)
	@rsync -a --delete --exclude .quarto/ --exclude _book/ --exclude __pycache__/ --exclude solutions/ \
	  --exclude '_quarto-website.yml' $(BOOK)/ $(STAGE_PDF)/
	@uv run python tools/terms.py build --book $(STAGE_PDF) --profile pdf

book: website pdf ## Build the website and the PDF into _book/

website: stage-website ## Build only the website (HTML, with the solutions) into _book/
	@$(call RENDER_CHECK,$(STAGE_WEB),website,html)
	@mkdir -p $(OUT)
	@rsync -a --delete --exclude '*.pdf' $(STAGE_WEB)/_book/ $(OUT)/
	@echo "Website: $(OUT)/index.html"

pdf: stage-pdf ## Build only the PDF (without the solutions)
	@$(call RENDER_CHECK,$(STAGE_PDF),pdf,pdf)
	@mkdir -p $(OUT)
	@cp $(STAGE_PDF)/_book/$(notdir $(PDF)) $(PDF)
	@echo "PDF: $(PDF)"

preview: stage-website ## Serve the website from the staged copy (rerun after editing book/)
	$(QUARTO) preview $(STAGE_WEB) --profile website

check-captions: stage-website ## Fail if any table or figure lacks a caption (or the book fails to render)
	@$(call RENDER_CHECK,$(STAGE_WEB),website,html)
	@echo "All tables and figures have captions."

format: ## Format all Python with ruff, as VSCode and the pre-commit hook do
	uv run ruff check --fix --exit-zero --quiet .
	uv run ruff format .

test: ## Run the tests
	uv run pytest

typecheck: ## Run strict pyright
	uv run pyright

check: test typecheck terms-check check-captions ## Tests, type checks, acronym and caption checks

all: check book ## Check everything, then build the website and the PDF

clean: ## Remove build output, staged copies and caches
	rm -rf $(OUT) $(BUILD)
