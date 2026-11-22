"""Tools for images: save one with its description, find stored ones, and
queue them to be sent with the reply."""

from __future__ import annotations

from typing import Any

from loguru import logger

from .context import Tool, ToolContext, index_written

# image_search: how many candidates the agent chooses from, and how much of
# each description it sees.
_IMAGE_CANDIDATES = 5
_IMAGE_DESCRIPTION_CHARS = 300

IMAGE_SAVE: dict[str, Any] = {
    "name": "image_save",
    "description": (
        "Save an image with your description so it can be retrieved later. "
        "You MUST call this when you receive an image with a media_ref in "
        "the prompt, or when the user shares a local image path. Describe "
        "the image in detail and pass the media_ref verbatim."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "Detailed image description"},
            "media_ref": {
                "type": "string",
                "description": "Opaque reference (file_id, local path, etc.) from the prompt",
            },
            "caption": {"type": "string", "description": "Optional caption"},
        },
        "required": ["description", "media_ref"],
    },
}

IMAGE_SEARCH: dict[str, Any] = {
    "name": "image_search",
    "description": (
        "Find previously stored images when the user asks to retrieve, show, "
        "or send one. Returns the closest matches with their ids and "
        "descriptions; nothing is sent. Then call image_send with the ids of "
        "the images that are what the user asked for."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query for images"},
        },
        "required": ["query"],
    },
}

IMAGE_SEND: dict[str, Any] = {
    "name": "image_send",
    "description": (
        "Send stored images to the user, by the ids image_search returned. "
        "Pass only the images that match the request (e.g. just the passport "
        "when the user asks for their passport)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Image ids from image_search",
            },
        },
        "required": ["ids"],
    },
}


async def image_save(ctx: ToolContext, args: dict) -> str:
    description: str = args["description"]
    media_ref: str = args["media_ref"]
    caption: str = args.get("caption", "")
    platform = ctx.platform or "local"

    combined = f"Image: {description}"
    if caption:
        combined += f" Caption: {caption}"
    # Register the image first: it needs an embedding, and if that fails
    # nothing has been written yet, so a retry doesn't save the note twice.
    await ctx.index.store_platform_image(
        platform=platform,
        media_ref=media_ref,
        description=combined,
        caption=caption,
    )
    store_path = ctx.store.save(combined, entry_type="image")
    await index_written(ctx, store_path)
    logger.info(
        "  → image_save({platform}) result: {desc}",
        platform=platform, desc=description[:100],
    )
    return f"Image saved: {description[:100]}"


async def image_search(ctx: ToolContext, args: dict) -> str:
    # Only images saved on this platform can be sent back on it (a
    # Telegram file_id means nothing to Slack). The terminal has no
    # platform and can describe images from anywhere.
    candidates = await ctx.search.search_images(
        args["query"], limit=_IMAGE_CANDIDATES, platform=ctx.platform,
    )
    logger.info("  → image_search result: {n} candidate(s)", n=len(candidates))
    if not candidates:
        return "No matching images found."
    lines = []
    for r in candidates:
        description = " ".join(r["description"].split())
        if len(description) > _IMAGE_DESCRIPTION_CHARS:
            description = description[:_IMAGE_DESCRIPTION_CHARS] + "…"
        lines.append(f"- id {r['id']} (match {r['score']:.2f}): {description}")
    return (
        "Closest stored images, best first (nothing sent yet):\n"
        + "\n".join(lines)
        + "\nCall image_send with the ids of the ones the user asked for."
    )


async def image_send(ctx: ToolContext, args: dict) -> str:
    ids = [int(i) for i in args.get("ids") or []]
    images = ctx.index.get_platform_images(ids, platform=ctx.platform)
    queued = {img["id"] for img in ctx.found_images}
    new = [img for img in images if img["id"] not in queued]
    ctx.found_images.extend(new)
    logger.info("  → image_send: {n} image(s) queued", n=len(new))
    missing = sorted(set(ids) - {img["id"] for img in images})
    note = f" No stored image with id {', '.join(map(str, missing))}." if missing else ""
    if not images:
        return "Nothing to send." + note
    return f"Sending {len(images)} image(s) with your reply." + note


TOOLS = [
    Tool(IMAGE_SAVE, image_save, "Saving the image"),
    Tool(IMAGE_SEARCH, image_search, "Searching images", "query"),
    Tool(IMAGE_SEND, image_send, "Sending images"),
]
