"""Small labelled-point helpers shared by the notebook's player and team charts."""

from textwrap import fill

import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image


def leading_player_labels(players, metric, extra_ids=(), count=3):
    """Include the metric leaders and explicitly requested native player IDs once."""
    selected = set(players.nlargest(count, metric, keep="all").fpl_player_id) | set(extra_ids)
    return players.loc[players.fpl_player_id.isin(selected)].copy()


def annotate_points(ax, frame, x, y, label="name", key="fpl_player_id", aliases=None,
                    offsets=None, above=False):
    """Attach transparent labels, optionally above points without connectors."""
    aliases, offsets = aliases or {}, offsets or {}
    annotations, occupied = [], []
    candidates = [(7, 7), (-7, 7), (7, -12), (-7, -12), (20, 20), (-20, 20),
                  (25, -25), (-25, -25), (45, 5), (-45, 5), (10, 35), (-10, -35)]
    if above:
        candidates = [(0, height, align) for height in range(7, 106, 14)
                      for align in ("center", "right", "left")]
        frame = frame.sort_values(y, kind="stable")
    else:
        candidates = [(dx, dy, "right" if dx < 0 else "left") for dx, dy in candidates]
    padding = .5 if above else 2
    ax.figure.canvas.draw()
    for row in frame.to_dict("records"):
        identifier = row[key]
        dx, dy = offsets.get(identifier, candidates[0][:2])
        align = "center" if above else "right" if dx < 0 else "left"
        text = fill(aliases.get(identifier, row[label]), width=18, break_long_words=False)
        annotation = ax.annotate(
            text, xy=(row[x], row[y]), xytext=(dx, dy), textcoords="offset points",
            ha=align,
            va="top" if dy < 0 else "bottom",
            fontsize=9.5, annotation_clip=False,
            bbox={"facecolor": "none", "edgecolor": "none", "pad": 1},
            arrowprops=None if above else {
                "arrowstyle": "-", "color": ".35", "linewidth": .6,
            },
        )
        annotation.set_in_layout(False)
        best, best_score = (dx, dy, align), float("inf")
        for dx, dy, align in [(dx, dy, align), *candidates]:
            annotation.set_position((dx, dy))
            annotation.set_ha(align)
            annotation.set_va("top" if dy < 0 else "bottom")
            ax.figure.canvas.draw()
            box = annotation.get_bbox_patch().get_window_extent().padded(padding)
            bounds = ax.get_window_extent()
            outside = max(bounds.x0 - box.x0, 0) + max(box.x1 - bounds.x1, 0)
            outside += max(bounds.y0 - box.y0, 0) + max(box.y1 - bounds.y1, 0)
            score = outside + 1000 * sum(box.overlaps(other) for other in occupied)
            if score < best_score:
                best, best_score = (dx, dy, align), score
            if score == 0:
                break
        annotation.set_position(best[:2])
        annotation.set_ha(best[2])
        annotation.set_va("top" if best[1] < 0 else "bottom")
        ax.figure.canvas.draw()
        occupied.append(annotation.get_bbox_patch().get_window_extent().padded(padding))
        annotations.append(annotation)
    ax.scatter(frame[x], frame[y], facecolors="none", edgecolors=".2", s=55, linewidths=.7)
    return annotations


def team_logo_paths(teams, logo_dir):
    """Resolve one season's native club IDs exclusively from the local logo folder."""
    if teams.season.nunique() != 1 or teams.fpl_team_id.duplicated().any():
        raise ValueError("Logo IDs must be unique within one season")
    filenames = {
        "Arsenal": "Arsenal FC.png", "Aston Villa": "Aston Villa.png",
        "Bournemouth": "AFC Bournemouth.png", "Brentford": "Brentford FC.png",
        "Brighton": "Brighton & Hove Albion.png", "Burnley": "Burnley FC.png",
        "Chelsea": "Chelsea FC.png", "Coventry City": "Coventry City.png",
        "Crystal Palace": "Crystal Palace.png", "Hull City": "Hull City.png",
        "Ipswich": "Ipswich Town.png", "Ipswich Town": "Ipswich Town.png",
        "Everton": "Everton FC.png", "Fulham": "Fulham FC.png", "Leeds": "Leeds United.png",
        "Leicester": "Leicester City.png", "Luton": "Luton Town.png",
        "Liverpool": "Liverpool FC.png", "Man City": "Manchester City.png",
        "Man Utd": "Manchester United.png", "Newcastle": "Newcastle United.png",
        "Nott'm Forest": "Nottingham Forest.png", "Spurs": "Tottenham Hotspur.png",
        "Sheffield Utd": "Sheffield United.png", "Southampton": "Southampton FC.png",
        "Sunderland": "Sunderland AFC.png", "West Ham": "West Ham United.png",
        "Wolves": "Wolverhampton Wanderers.png",
    }
    result = {}
    for row in teams.itertuples(index=False):
        if row.team not in filenames:
            raise ValueError(f"No logo filename configured for {row.team}")
        path = logo_dir / filenames[row.team]
        if not path.is_file():
            raise FileNotFoundError(f"Missing local logo for {row.team}: {path}")
        result[row.fpl_team_id] = path
    return result


def annotate_team_logos(ax, frame, x, y, paths, size_points=13):
    """Use aspect-correct crests as markers centred on exact data coordinates."""
    artists = []
    for row in frame.to_dict("records"):
        with Image.open(paths[row["fpl_team_id"]]) as source:
            rgba = source.convert("RGBA")
            image = np.asarray(rgba.crop(rgba.getchannel("A").getbbox()))
        crest = OffsetImage(image, zoom=size_points / max(image.shape[:2]))
        artist = AnnotationBbox(
            crest, (row[x], row[y]), xycoords="data", frameon=False, pad=0,
        )
        artist.set_in_layout(False)
        ax.add_artist(artist)
        artists.append(artist)
    ax.update_datalim(frame[[x, y]].to_numpy(dtype=float))
    ax.autoscale_view()
    return artists
