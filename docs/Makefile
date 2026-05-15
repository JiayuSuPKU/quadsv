# Minimal makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help html clean livehtml

help:
	@echo "Available targets:"
	@echo "  html      Build HTML documentation"
	@echo "  clean     Remove build artifacts"
	@echo "  livehtml  Start auto-rebuilding dev server (requires sphinx-autobuild)"

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS)
	@echo
	@echo "Build finished. Open $(BUILDDIR)/html/index.html to view."

clean:
	rm -rf $(BUILDDIR)

livehtml:
	sphinx-autobuild $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS)
