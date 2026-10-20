# Public release checklist

- Run python verify_checksums.py before regeneration.
- Run python reproduce.py, then python -m unittest discover -s tests -v.
- Check results/validation.json and the generated tables.
- Approve licences and third-party redistribution, and verify CITATION.cff authorship.
- Publish the reviewed repository release and archive that exact version with a persistent identifier.
- Insert the real repository URL and version DOI in CITATION.cff and Data Availability.

No external publication has been performed by preparation of this package.
