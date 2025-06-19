# Makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = .
BUILDDIR     = _build

# O is used as a shortcut for additional Sphinx options
O ?=

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Enhanced build targets
clean:
	@echo "Cleaning build directory..."
	rm -rf $(BUILDDIR)/*
	rm -rf _autosummary/
	@echo "Clean completed."

html: clean
	@echo "Building HTML documentation..."
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

html-fast:
	@echo "Building HTML documentation (incremental)..."
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

pdf:
	@echo "Building PDF documentation..."
	$(SPHINXBUILD) -b latex "$(SOURCEDIR)" "$(BUILDDIR)/latex" $(SPHINXOPTS) $(O)
	@make -C $(BUILDDIR)/latex all-pdf
	@echo "PDF build finished. The PDF files are in $(BUILDDIR)/latex."

epub:
	@echo "Building EPUB documentation..."
	$(SPHINXBUILD) -b epub "$(SOURCEDIR)" "$(BUILDDIR)/epub" $(SPHINXOPTS) $(O)
	@echo "EPUB build finished. The EPUB file is in $(BUILDDIR)/epub."

linkcheck:
	@echo "Checking external links..."
	$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS) $(O)
	@echo "Link check complete; look for any errors in the above output."

doctest:
	@echo "Running doctests..."
	$(SPHINXBUILD) -b doctest "$(SOURCEDIR)" "$(BUILDDIR)/doctest" $(SPHINXOPTS) $(O)
	@echo "Doctest finished."

coverage:
	@echo "Checking documentation coverage..."
	$(SPHINXBUILD) -b coverage "$(SOURCEDIR)" "$(BUILDDIR)/coverage" $(SPHINXOPTS) $(O)
	@echo "Coverage check finished; see $(BUILDDIR)/coverage/python.txt."

serve:
	@echo "Starting documentation server on http://localhost:8000"
	@cd $(BUILDDIR)/html && python -m http.server 8000

# Development targets
dev: html-fast serve

autobuild:
	@echo "Starting auto-building documentation server..."
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O) \
		--host 0.0.0.0 --port 8000 --watch ../src

# Quality checks
check: linkcheck doctest coverage

# Full rebuild with all checks
all: clean html pdf epub check
	@echo "Full documentation build completed."

# Generate API documentation from source
apidoc:
	@echo "Generating API documentation from source..."
	sphinx-apidoc -f -o . ../src
	@echo "API documentation generated."

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
