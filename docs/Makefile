# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
DEPLOYDIR     = _deploy
GH_PAGES_REPO = git@github.com:educating-dip/docs.bayes_dip.git

# for apidoc
PYTHONSOURCEDIR	= ../
APIDOCDIR       = _apidoc
EXCLUDE = ""
EXCLUDE += $(shell git ls-files -z --others --exclude-standard $(PYTHONSOURCEDIR) | xargs -0)

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

apidoc:
	@sphinx-apidoc -f -e -o "$(APIDOCDIR)" "$(PYTHONSOURCEDIR)" $(EXCLUDE) --no-toc

clean:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	rm -rf $(APIDOCDIR)/*.rst
	rm -rf $(DEPLOYDIR)

deploy-gh-pages: html
	rm -rf $(DEPLOYDIR)/gh-pages
	git clone $(GH_PAGES_REPO) $(DEPLOYDIR)/gh-pages
	rm -rf $(DEPLOYDIR)/gh-pages/*
	cp -r $(BUILDDIR)/html/* $(DEPLOYDIR)/gh-pages/
	cd $(DEPLOYDIR)/gh-pages && git add . && git commit -m "Update docs" && git push || echo "nothing to commit"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
# Always run apidoc beforehand.
%: Makefile apidoc
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
