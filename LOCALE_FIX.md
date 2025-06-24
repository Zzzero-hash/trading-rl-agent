# Locale Fix for Git Commit Issues

## Problem

The trading-rl-agent project was experiencing git commit warnings:

```
bash: warning: setlocale: LC_ALL: cannot change locale (en_US.UTF-8)
```

## Root Cause

The dev container environment was configured to use `C.UTF-8` locale, but the system only has `C.utf8` available (lowercase 'utf8').

## Solution

1. **Updated locale settings** in all environment scripts to use `C.utf8`
2. **Added global locale configuration** in `.env.locale`
3. **Updated user profile** to persist locale settings
4. **Enhanced dev-setup.sh** to automatically configure locale

## Files Changed

- `dev-setup.sh` - Updated locale and added auto-configuration
- `setup-env.sh` - Fixed locale settings
- `.env.locale` - New global locale configuration file
- `~/.bashrc` - Added locale exports

## Usage

Run the development setup script to apply locale fixes:

```bash
./dev-setup.sh
```

Or source it for immediate effect:

```bash
source ./dev-setup.sh
```

## Verification

Check that locale is properly set:

```bash
locale
```

Test git commit without warnings:

```bash
git commit -m "test commit"
```

## Troubleshooting

If you still see locale warnings:

1. Check available locales: `locale -a`
2. Restart your terminal/shell
3. Re-run the dev-setup script
4. Verify environment variables: `echo $LC_ALL`
