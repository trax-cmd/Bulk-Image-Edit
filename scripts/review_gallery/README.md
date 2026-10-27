# Review gallery

Builds the browsable "TraxNYC Image Review" page: every processed image as an
ORIGINAL | CORRECTED pair packed into WebP sprite sheets (32 pairs per sheet,
400 px tiles), plus a `data.js` index with what was removed, the QA severity
and note, and whether the image was re-processed after that note.

```
S=<session scratchpad root> python3 -I scripts/review_gallery/build_gallery.py
```

Inputs (under `$S`): `out/report_v3_final2.csv`, `deliver_v3/images_to_review.csv`,
originals in `drive/files/<item>/<file>`, outputs in `out/all_v3/<item>/<stem>.jpg`.
Output: `$S/gallery/{index.html,data.js,data.json,sprites/sNNN.webp}`.

`index.html` is the page itself (published as a claude.ai artifact with the
`db` and `user` capabilities so the reviewer's flags are stored server-side in
the `flags` collection, keyed by a hash of item/file, and the scroll progress
in `progress/main`).
