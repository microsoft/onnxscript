final_readme = []
with open('README.md', encoding="utf-8") as f:
    readme = f.readlines()
    for line in readme:
        final_readme.append(line)
        if line == "```python update-readme\n":
            with open('docs/tutorial/examples/hardmax_end_to_end.py', encoding="utf-8") as example_f:
                example_code = example_f.readlines()
                final_readme += example_code

with open('README.md', 'w', encoding="utf-8") as f:
    f.writelines(final_readme)
