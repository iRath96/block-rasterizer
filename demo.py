from rasterizer import Rasterizer

renderer = Rasterizer(
    "/Users/alex/Library/Application Support/minecraft/versions/20w19a/20w19a.jar",
    (128, 128)
)

# supports transparency
renderer.render_block(
    "beacon"
).save("demo/beacon.png")

# supports rotations
renderer.render_block(
    "lectern",
    "facing=south"
).save("demo/lectern.png")

# supports multipart models
renderer.render_block(
    "brewing_stand",
    "has_bottle_0=true,has_bottle_1=false,has_bottle_2=false"
).save("demo/brewing_stand.png")

# and even more blocks...

renderer.render_block(
    "comparator",
    "facing=west,mode=subtract,powered=false"
).save("demo/comparator.png")

renderer.render_block(
    "campfire",
    "facing=north,lit=true"
).save("demo/campfire.png")

renderer.render_block(
    "tnt"
).save("demo/tnt.png")
