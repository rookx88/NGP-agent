# Assets Directory

This folder contains static assets for the Synch frontend UI.

## Structure

- `images/` — All SVGs, PNGs, and other image assets
  - `icons/` — SVG icon files (e.g., trending.svg, influencers.svg)
  - `backgrounds/` — SVG or PNG backgrounds (e.g., gradient-bg.svg)

## Usage
- Import SVGs as React components or image URLs in your components:
  - `import { ReactComponent as Logo } from './images/logo.svg';`
  - `import trendingIcon from './images/icons/trending.svg';`

Place your SVGs and other images in the appropriate subfolders for easy access and organization.

