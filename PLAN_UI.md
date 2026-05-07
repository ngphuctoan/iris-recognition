# Implementation Plan: SSR Biometric Terminal (v7.0)

This plan outlines the final transition from approved mockups in `temp/` to a production-grade, SSR-only FastAPI application using the "Pure No-Nonsense" design system.

## 1. Architectural Foundation
- **Framework**: FastAPI + Jinja2 Templates.
- **SSR Strategy**: Full-page reloads for all transitions and form submissions.
- **CSS Strategy**: Tailwind CSS v4 CDN with a unified Design Token block in `base.html`.
- **Logic**: No client-side Javascript allowed.

## 2. Template Architecture
- **`templates/base.html`**:
  - Global `<head>` with Overpass fonts and RemixIcon CDN.
  - Sticky Top Header with horizontal navigation.
  - Responsive layout shell (`main` container).
  - Design Token block (colors, custom utility classes).
- **`templates/identify.html`**:
  - Match Pipeline Input (Dashed Upload Row).
  - Primary Match Result (Conditional block: Only shown if results exist).
  - Diagnostic Metrics Grid.
- **`templates/enroll.html`**:
  - Metadata form.
  - Dataset Sync form.
  - Manual Capture form.
- **`templates/directory.html`**:
  - High-density registry table.
  - Empty state handling.

## 3. Design Token Implementation (CSS)
All templates will inherit these tokens from `base.html`:
- **Colors**: Bone White (`#fcfcfc`), Deep Royal Blue (`#004085`), Obsidian (`#2d3436`).
- **Typography**: `Overpass` (Sans/Mono).
- **Constraints**: 
  - `max-w-none` on all containers.
  - Padding/Margin capped at `4` (`1rem`).
  - Icons fixed at `20px`.
  - Font floor at `10px`.
  - Max font size capped at `2xl`.

## 4. API & Routing Tasks
- [ ] Refactor `src/api/main.py` to serve `TemplateResponse`.
- [ ] Implement `GET /` (Redirect to `/identify`).
- [ ] Implement `GET /identify` and `POST /identify` (Handling file uploads).
- [ ] Implement `GET /enroll` and `POST /enroll` (Handling registration).
- [ ] Implement `GET /directory` (Querying the subject database).

## 5. Verification Checkpoints
- [ ] Responsive test: Verify 2-column to 4-column grid shift on Diagnostics.
- [ ] Semantic check: Verify all text uses role-based typescale.
- [ ] Constraint check: Verify no padding exceeds `p-4` and icons are `20px`.
