# Push to GitHub — two commands

**The git history is already made.** This folder is a git repository with every
file committed on the `main` branch. You do not need `git init`, `git add` or
`git commit`. Two commands and it is up.

I could not create the repository for you: the sandbox I run in can read your
GitHub identity but is blocked from creating repositories or pushing to them.
Step 1 has to be done by you.

---

## 1. Create the empty repo (1 minute)

1. Go to **github.com/new**
2. Repository name: **`AlgorithmX`**
3. Description: *SAMUDRA NETRA — satellite oil-spill detection and vessel attribution. SIH 2026, SIH26143.*
4. **Public**
5. **Do not tick** "Add a README", "Add .gitignore" or "Choose a license". This
   folder already has all three, and ticking any of them makes your first push
   fail with `failed to push some refs`.
6. **Create repository**

---

## 2. Push (2 minutes)

Open Terminal, type `cd ` (with a space), drag **this folder** from Finder into
the Terminal window, press Enter. Then:

```bash
git remote add origin https://github.com/yagnadeepreddy081-reddy/AlgorithmX.git
git push -u origin main
```

It will ask for a username and password.

- Username: `yagnadeepreddy081-reddy`
- Password: **not your GitHub password** — GitHub stopped accepting those. You
  need a token, below.

### Making the token

1. github.com → your avatar → **Settings**
2. Bottom of the left menu → **Developer settings**
3. **Personal access tokens → Tokens (classic) → Generate new token (classic)**
4. Note: `AlgorithmX push`. Expiration: 30 days. Tick the **`repo`** checkbox.
5. **Generate token**, then copy it — it is shown once and never again.
6. Paste it as the password. Nothing appears on screen while you paste. That is
   normal. Press Enter.

Refresh the GitHub page and all 45 files will be there.

---

## 3. Host it from the repo (optional, 2 minutes)

1. **vercel.com** → **Add New** → **Project** → **Import** `AlgorithmX`
2. Framework preset: **Other**
3. **Root Directory: `public`** ← get this wrong and you get a blank page
4. **Deploy**

You get a URL like `https://algorithmx.vercel.app`. Put it at the top of
`README.md` in place of the placeholder link, then:

```bash
git add README.md
git commit -m "Add live demo link"
git push
```

Every push from then on redeploys the site automatically.

---

## What a judge sees

| They click | They find |
|---|---|
| Repo front page | README: the idea, the real numbers, and an honest list of limitations |
| `src/` | 18 Python modules, one per pipeline stage |
| `web/engine.js` | The U-Net forward pass written by hand in JavaScript |
| `results/` | The actual metrics files and the training log, not screenshots |
| **Actions** tab | A green tick — the smoke test runs on every push |

---

## If it goes wrong

| Message | Cause and fix |
|---|---|
| `git: command not found` | Run `xcode-select --install`, accept, wait, retry |
| `failed to push some refs` | You ticked README/.gitignore/license when creating the repo. Run `git pull --rebase origin main` then push again |
| `Support for password authentication was removed` | You used your password. Make a token, above |
| `remote origin already exists` | Run `git remote set-url origin https://github.com/yagnadeepreddy081-reddy/AlgorithmX.git` then push |
| Vercel deploys but the page is blank | Root Directory was not set to `public`. Project → Settings → General → Root Directory → `public` → redeploy |

---

## A note on the repo name

The repository is **AlgorithmX**; the project inside it is **SAMUDRA NETRA**.
That is a normal arrangement — a team name on the outside, a product name on the
inside — and the README makes the project name obvious on the first line. If you
would rather they match, rename the repo in Settings → General; nothing in the
code depends on the repository name.
