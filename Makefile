# Compatibility shortcuts; run from docs/ with the docs dependency group.
.PHONY: html serve clean
html:
	cd .. && uv run --locked --only-group docs mkdocs build --strict
serve:
	cd .. && uv run --locked --only-group docs mkdocs serve
clean:
	$(RM) -r ../site
