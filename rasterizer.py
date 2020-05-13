import zipfile
import json
import numpy
from io import BytesIO
from PIL import Image, ImageEnhance, ImageOps, ImageDraw

class JarFileProvider(object):
    def __init__(self, jar_path):
        self.jar = zipfile.ZipFile(jar_path)
    
    def find_file(self, filename):
        return self.jar.open(filename)

    def load_json(self, filename):
        fileobj = self.find_file(filename)
        data = json.load(fileobj)
        return data

    def load_image(self, filename):
        fileobj = self.find_file(filename)
        buffer = BytesIO(fileobj.read())
        img = Image.open(buffer).convert("RGBA")
        return img

class Rasterizer(object):
    def __init__(self, file_provider, dimensions=(24, 24)):
        self.file_provider = file_provider
        self.bgcolor = (26, 26, 26, 0)
        self.texture_dimensions = dimensions

        self.light_vector  = numpy.array([ -0.8, +1.0, +0.7 ])
        #self.light_vector /= numpy.linalg.norm(self.light_vector)

        self.projection = numpy.array([
            [ +1/2,    0, +1/2 ],
            [ -1/4, -1/2, +1/4 ],
            [    0,    1,    0 ]
        ])

        self.global_matrix = numpy.eye(3)
        #self.global_matrix = self.rotation_matrix("y", 180)

        self.cube_faces = {
            "up":    [ numpy.array(v) for v in ([0,1,0], [1,1,0], [0,1,1]) ],
            "down":  [ numpy.array(v) for v in ([0,0,1], [1,0,1], [0,0,0]) ],
            "north": [ numpy.array(v) for v in ([1,1,0], [0,1,0], [1,0,0]) ],
            "west":  [ numpy.array(v) for v in ([0,1,0], [0,1,1], [0,0,0]) ],
            "south": [ numpy.array(v) for v in ([0,1,1], [1,1,1], [0,0,1]) ],
            "east":  [ numpy.array(v) for v in ([1,1,1], [1,1,0], [1,0,1]) ],
        }

        # Create a grid of sampling positions
        width, height = self.texture_dimensions
        xs = numpy.linspace(0, 1, width, endpoint=False) + 0.5/width
        ys = numpy.linspace(0, 1, height, endpoint=False) + 0.5/height
        xpos = numpy.tile(xs, (height, 1)).transpose()
        ypos = numpy.tile(ys, (width, 1))

        # Reshape the sampling positions to a H x W x 2 tensor
        pos = numpy.moveaxis(numpy.array(list(zip(ypos, xpos))), 1, 2)
        pos = numpy.reshape(pos, (height*width, 2))

        self.img_grid = pos

    def remove_namespace(self, name):
        # TODO check if this starts with a different namespace
        if name[0:10] == "minecraft:":
            name = name[10:]
        return name

    def render_quad(self, image, zbuffer, texture, uv, uv_matrix, points):
        # extended version of https://gist.github.com/seece/4b170e21ccd3aa12e747b7702464a727

        shift = numpy.array([0, 3/4, 1/4])
        d = numpy.array([[
            numpy.matmul(self.projection, point) + shift for point in points
        ]])

        normal  = numpy.cross(points[2] - points[0], points[1] - points[0])
        normal /= numpy.linalg.norm(normal)
        light = numpy.clip(normal.dot(self.light_vector), 0, 1)

        def edgefunc(v0, v1, p):
            px = p[:, 1]
            py = p[:, 0]
            return (v0[:,0] - v1[:,0]) * px + (v1[:,1] - v0[:,1]) * py + (v0[:,1] * v1[:,0] - v0[:,0] * v1[:,1])

        area = edgefunc(d[:,2,:], d[:,1,:], d[:,0,:])[0]
        if area <= 0:
            # back-face culling (i.e., we're looking at the backside of the triangle)
            return False

        # evaluate the edge functions at every position
        w0 = edgefunc(d[:,1,:], d[:,0,:], self.img_grid) / area
        w1 = edgefunc(d[:,0,:], d[:,2,:], self.img_grid) / area

        # calculate texture coordinates
        texcoords = numpy.matmul(uv_matrix, numpy.stack([ w1, w0 ]) - 0.5) + 0.5

        # map to pixel coordinates
        upix = numpy.clip((uv[0] + texcoords[0,:] * (uv[2] - uv[0])).astype(numpy.int), 0, texture.shape[1]-1)
        vpix = numpy.clip((uv[1] + texcoords[1,:] * (uv[3] - uv[1])).astype(numpy.int), 0, texture.shape[0]-1)
        tex = texture[vpix,upix,:]

        # calculate depth
        z = (1 - w0 - w1) * d[0,0,2] + w0 * d[0,2,2] + w1 * d[0,1,2]

        # find pixels to be overwritten
        mask = (w0 >= 0) & (w1 >= 0) & (w0 < 1) & (w1 < 1) # only within quad
        mask = mask & (z > zbuffer) # only foreground
        mask = mask & (tex[:,3] > 0) # only opaque pixels
        
        image[mask] = tex[mask] * light
        image[:,3][mask] = 255
        zbuffer[mask] = z[mask]

        return True

    def rotation_matrix(self, axis, degree):
        angle = degree / 180 * numpy.pi
        cos, sin = numpy.cos(angle), numpy.sin(angle)
        axis0 = { "x": 0, "y": 1, "z": 2 }[axis]
        axis1 = (axis0 + 1) % 3
        axis2 = (axis0 + 2) % 3

        matrix = numpy.zeros((3, 3))
        matrix[axis0,axis0] = 1
        matrix[axis1,axis1] = cos
        matrix[axis1,axis2] = -sin
        matrix[axis2,axis1] = +sin
        matrix[axis2,axis2] = cos

        return matrix

    def render_block(self, block_name, variant="", props={}):
        image = numpy.zeros((
            self.texture_dimensions[0] * self.texture_dimensions[1],
            4
        ), dtype=numpy.uint8)

        zbuffer = numpy.zeros((
            self.texture_dimensions[0] * self.texture_dimensions[1]
        ))

        blockstates = self.file_provider.load_json(
            "assets/minecraft/blockstates/%s.json" % block_name
        )

        if "variants" in blockstates:
            blockstate = blockstates["variants"][variant]
            self.render_part(blockstate, image, zbuffer, props)
        elif "multipart" in blockstates:
            state = dict((x.split("=") for x in variant.split(",")))
            for part in blockstates["multipart"]:
                if all((state[k] == v for (k,v) in part.get("when", {}).items())):
                    self.render_part(part["apply"], image, zbuffer, props)

        image = image.reshape((*self.texture_dimensions, 4))
        return Image.fromarray(image)
        
    def render_part(self, blockstate, image, zbuffer, props):
        if isinstance(blockstate, list):
            # @todo why?
            self.render_part(blockstate[0], image, zbuffer, props)
            return

        global_matrix = self.global_matrix
        for axis in ["x", "y", "z"]:
            if axis in blockstate:
                global_matrix = numpy.matmul(
                    global_matrix,
                    self.rotation_matrix(axis, -blockstate[axis])
                )
        global_shift = numpy.matmul(
            numpy.eye(3) - global_matrix,
            numpy.ones((3)) * 8
        )[:,numpy.newaxis]

        textures = {}
        def resolve_texture(v):
            return (textures[v[1:]] if v[0] == "#" else v)
        
        model_name = blockstate["model"]
        while True:
            model = self.file_provider.load_json(
                "assets/minecraft/models/%s.json" % self.remove_namespace(model_name)
            )

            if "textures" in model:
                textures.update({
                    k: resolve_texture(v)
                    for k, v in model["textures"].items()
                })
            if "elements" in model:
                break
            if "parent" in model:
                model_name = model["parent"]
            else:
                break

        if model_name == "block/cube":
            props["solid"] = True

        if "elements" not in model:
            raise KeyError("'%s' does not have a model" % model_name)

        for element in model["elements"]:
            points = numpy.array([ element["from"], element["to"] ])

            # elements can have custom rotation
            local_matrix, local_shift = numpy.eye(3), 0
            if "rotation" in element:
                axis, angle = element["rotation"]["axis"], element["rotation"]["angle"]
                local_matrix = self.rotation_matrix(axis, angle)
                local_shift = numpy.matmul(
                    numpy.eye(3) - local_matrix,
                    numpy.array(element["rotation"]["origin"])
                )[:,numpy.newaxis]
            
            for side_name, side_data in element["faces"].items():
                quad = self.cube_faces[side_name]
                quad = numpy.array([ numpy.diag(points[v]) for v in quad ]).transpose()
                uv = side_data.get("uv", None)
                if not uv:
                    # taken from https://github.com/DragonDev1906/Minecraft-Overviewer/
                    _from, _to = element["from"], element["to"]
                    uv = {
                        "north": (_to  [0], 16-_to[1], _from[0], 16-_from[1]),
                        "east":  (_from[2], 16-_to[1], _to  [2], 16-_from[1]),
                        "south": (_from[0], 16-_to[1], _to  [0], 16-_from[1]),
                        "west":  (_from[2], 16-_to[1], _to  [2], 16-_from[1]),
                        "up":    (_from[0], _from [2], _to  [0], _to     [2]),
                        "down":  (_to  [0], _from [2], _from[0], _to     [2]),
                    }[side_name]

                # apply transformations
                quad = numpy.matmul(local_matrix, quad) + local_shift
                quad = numpy.matmul(global_matrix, quad) + global_shift
                quad = quad.transpose() / 16 # scale down to [0;1] range

                texture = numpy.asarray(self.file_provider.load_image(
                    "assets/minecraft/textures/%s.png" % self.remove_namespace(
                        resolve_texture(side_data["texture"])
                    )
                ))

                uv_rotation = side_data.get("rotation", 0)/180*numpy.pi
                cos, sin = numpy.cos(uv_rotation), numpy.sin(uv_rotation)
                uv_matrix = numpy.array([
                    [  cos, sin ],
                    [ -sin, cos ]
                ])
                
                self.render_quad(image, zbuffer, texture, uv, uv_matrix, quad)
