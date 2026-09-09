# Put this on Vercel — 4 minutes

You have two files in this folder: `index.html` and `vercel.json`. That is the
whole website. No build step, no framework, no database, no environment
variables.

I could not deploy it for you because Vercel's servers are blocked from the
sandbox I run in. But this is genuinely a 4-minute job, and you only need a
free Vercel account.

---

## Method A — the terminal (fastest, no GitHub needed)

You need Node.js installed. Check with `node -v`. If it prints a version, you
are fine; if not, install it from nodejs.org first.

Open a terminal **inside this folder** and run:

```bash
npx vercel login
```

Pick "Continue with GitHub" (or email). A browser tab opens, you approve, done.

Then:

```bash
npx vercel --prod
```

It will ask you a few questions. The answers are:

| Question | Answer |
|---|---|
| Set up and deploy? | **y** |
| Which scope? | your own name |
| Link to existing project? | **n** |
| What's your project's name? | `samudra-netra` |
| In which directory is your code located? | `./` (just press Enter) |
| Want to modify these settings? | **n** |

After about 30 seconds it prints a URL like:

```
https://samudra-netra.vercel.app
```

**That is your live site.** Open it, check it loads, and use that link tomorrow.

---

## Method B — GitHub, if you want the code visible to judges too

1. Create a new **public** repo on github.com, e.g. `samudra-netra`.
2. Push this folder to it:

```bash
git init
git add .
git commit -m "SAMUDRA NETRA - SIH26143 prototype"
git branch -M main
git remote add origin https://github.com/<your-username>/samudra-netra.git
git push -u origin main
```

3. Go to **vercel.com → Add New → Project → Import** your repo.
4. Framework preset: **Other**. Root directory: `./`. Press **Deploy**.

Every time you push, the site updates itself. This route looks more serious to
a technical judge because they can read the source, so if you have the extra
ten minutes, take it — and push the `src/` folder from the source zip into the
same repo so the Python is visible too.

---

## Renaming the URL

In the Vercel dashboard: your project → **Settings → Domains** → add a domain.
Anything free like `samudra-netra.vercel.app` or `samudranetra-sih.vercel.app`
works. Pick one you can say out loud without spelling it.

---

## Two things to check before you sleep

1. **Open the deployed URL on your phone, on mobile data, not college Wi-Fi.**
   If it loads there, it will load tomorrow.
2. **Keep `index.html` on a USB stick or in your laptop's Downloads.** If the
   venue internet dies completely, you open the file directly from disk and the
   demo is identical — the page has no network dependencies at all. This is your
   parachute. Do not skip it.

---

## If something goes wrong

- `npx vercel` says command not found → Node.js isn't installed.
- Deploy succeeds but the page is blank → you deployed the wrong folder; make
  sure `index.html` sits at the top level of what you deployed.
- Page loads slowly the first time → it is a 9 MB file with 41 radar images
  baked in. It caches after the first load. Open it once before you present.
