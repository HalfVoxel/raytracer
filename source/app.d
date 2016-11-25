/* This program prints a
   hello world message
   to the console.  */
 
import std.stdio;
import tga;
import gl3n.linalg;
import gl3n.interpolate;
import std.random;
import std.math;
import gl3n.ext.hsv;

void main()
{
    writeln("Hello, World!");

    ushort width = 1024;
    ushort height = 1024;
    auto pixelBitDepth = 8;

	/* generate pixels in some way */
	Pixel[] pixels = generateImage(width, height);

	Image img = createImage(pixels, width, height);//, ImageType.COMPRESSED_TRUE_COLOR, pixelBitDepth);

	File outFile = File("output.tga","w");
	//outFile.write("Hello");

	writeImage(outFile, img);
}

class Material {
	vec4 color;
	float reflectivity;
	float transparency;
	float spec;
	float emissive = 0;
}

class Light {
	vec3 position;
	vec4 color;

	this ( vec3 position, vec4 color, float intensity ) {
		this.position = position;
		this.color = color * intensity;
	}

	vec4 getColor ( vec3 point ) {
		float att = max(0,1.0f - 0.1f*(point-position).length);//1.0f / (point - position).magnitude_squared;
		return color * att;
	}
}

class Shape {
	vec3 position;
	Material mat;

	float distance ( vec3 origin, vec3 direction, out bool backface ) {
		backface = false;
		return 0;
	}

	void details ( vec3 origin, vec3 direction, float dist, vec3 point, out vec3 normal ) {
		normal = vec3(0,1,0);
	}

	float distance ( vec3 p ) {
		return 10000000;
	}

	vec3 normal ( vec3 p ) {
		return vec3(0,1,0);
	}
}

class Plane : Shape {

	vec3 normal;

	this ( vec3 position, vec3 normal ) {
		this.position = position;
		this.normal = normal;
	}

	override float distance ( vec3 origin, vec3 direction, out bool backface ) {
		backface = false;
		vec3 rel = position - origin;
		float t = dot ( normal, rel ) / dot ( normal, direction );
		return t;
	}

	override void details ( vec3 origin, vec3 direction, float dist, vec3 point, out vec3 normal ) {
		normal = this.normal;
	}

	override float distance ( vec3 p ) {
		return 0;
		//p0 dot v = 0
		//v dot (p-v) = 0
	}
}

class Sphere : Shape {
	float radius;

	this ( vec3 position, float radius ) {
		this.position = position;
		this.radius = radius;
	}

	override float distance ( vec3 origin, vec3 direction, out bool backface ) {
		vec3 offset = position - origin;
		float d1 = dot(offset, direction);
		float d2 = sqrt(offset.magnitude_squared - d1^^2);
		float d3 = sqrt(radius^^2 - d2^^2);

		float dist = (d1 - d3);
		float dist2 = (d1 + d3);

		backface = false;

		vec3 p = origin + direction*dist;
		if (cast(int)(p.y*20 + sin(p.x*12)*0.2f) % 2 == 0 ) {
			
			backface = true;
			dist = dist2;
			p = origin + direction*dist;
			if (cast(int)(p.y*20 + sin(p.x*12)*0.2f) % 2 == 0 ) {
				return float.infinity;
			}
		}
		//dist *= 1.0f + 0.3f * sin(p.x*3) * cos(p.y*3);

		return dist;
	}

	override void details ( vec3 origin, vec3 direction, float dist, vec3 point, out vec3 normal ) {
		normal = (point - position).normalized;
	}

	override float distance ( vec3 p ) {
		return max((this.position - p).magnitude - radius, 0);
	}

	override vec3 normal ( vec3 p ) {
		return (p - position).normalized;
	}
}

static vec3 reflect ( vec3 v, vec3 normal ) {
	vec3 p = dot(v,normal) * normal;
	return p + (p-v);
}

struct Settings {
	int maxDepth;
	int emissiveRays;

}

class Viewport {
	vec3 cameraPos;
	vec2 cameraSize;
	Scene scene;
}

class Renderer {
	void render(Viewport vp, ushort width, ushort height, vec4[] pixels) {}
}

class Scene {
	Shape[] shapes;
	Light[] lights;
	auto rnd = Random();

	vec4 ambLight = vec4(0.5f,0.5f,0.5f,0.0f);

	float intersection ( vec3 rayOrig, vec3 rayDir, float maxDist ) {
		float light = 1;

		foreach ( ref shape; shapes ) {
			bool bf;
			float d = shape.distance(rayOrig, rayDir, bf);
			if ( d < maxDist && d > 0.001f ) {
				light *= shape.mat.transparency;
			}
			//writeln(d);
		}
		return light;
	}

	vec4 getColor ( vec3 rayOrig, vec3 rayDir, int depth, out float distance, bool emissive = false ) {
		Shape firstShape;
		float firstZ = float.infinity;
		bool backface = false;

		vec4 c = vec4 (0,0,0,0);

		foreach ( ref shape; shapes ) {
			vec3 point;
			bool bf;
			float d = shape.distance(rayOrig, rayDir, bf);
			if ( d < firstZ && d > 0.001f ) {
				backface = bf;
				firstZ = d;
				firstShape = shape;
			}
			//writeln(d);
			
		}

		distance = firstZ;

		if ( firstShape is null ) {
			return vec4(0,0,0,0);
		}

		if ( depth > 3 ) {
			c = firstShape.mat.color;
			if ( emissive ) {
				c *= firstShape.mat.emissive;
			}
			return c;
		}

		vec3 point = rayOrig + rayDir * firstZ;

		vec3 normal;
		firstShape.details(rayOrig, rayDir, firstZ, point, normal );

		if ( backface ) {
			normal = -normal;
		}

		c = firstShape.mat.color;

		if ( firstShape.mat.reflectivity > 0 ) {
			auto newDir = reflect ( -rayDir, normal );
			float dummy;
			vec4 crefl = getColor ( point, newDir, depth+1, dummy);
			
			//writeln("Colors");
			//writeln(c);
			//writeln(crefl);
			c = lerp ( c, crefl, firstShape.mat.reflectivity );
			//writeln(c);
		}

		if ( firstShape.mat.transparency > 0 ) {
			float dummy;
			vec4 ctrans = getColor ( point, rayDir, depth+1, dummy );

			
			c = lerp(c,ctrans, firstShape.mat.transparency);
		}

		vec4 res = vec4(0,0,0,0);

		foreach (ref light; lights) {
			vec3 ld = (light.position - point);
			float magn = ld.length;
			vec3 ldo = ld;
			ld = ld.normalized;
			float reached = intersection (point, ld, magn ); 
			//res = vec4(1,1,1,1)*reached;
			if ( reached > 0 ) {
				float shading;

				vec4 lightCol = light.getColor (point) * reached;

				if ( firstShape.mat.spec > 0) {
					vec3 h = (ld + (-rayDir)).normalized;

			        float diff = max (0, dot (normal, ld));

			        float nh = max (0, dot (normal, h));
			        float spec = nh^^48;

			        vec4 c1 = (lightCol * diff);
					c1.r *= c.r;
					c1.g *= c.g;
					c1.b *= c.b;
					c1.a *= c.a;
					c1 += lightCol * spec * firstShape.mat.spec;

					res += c1;
				} else {
					shading = dot(ld, normal);
					vec4 c1 = lightCol * max(0,shading);
					c1.r *= c.r;
					c1.g *= c.g;
					c1.b *= c.b;
					c1.a *= c.a;
					res += c1;
				}
				
			} else {
			}
		}

		if ( depth == -1 ) {
			vec4 emCol = vec4(0,0,0,0);

			foreach (i; 0..30) {
				vec3 dir = vec3(uniform(-1f,1f),uniform(-1f,1f),uniform(-1f,1f)).normalized;
				float dist;
				vec4 col = getColor (point, dir, depth+5, dist, true);
				emCol += col * (1.0f/(1*dist+1));
			}
			emCol *= 1.0f/20.0f;
			emCol *= 1;
			//writeln(emCol);
			c += emCol;
		}

		vec4 c2 = ambLight;
		c2.r *= c.r;
		c2.g *= c.g;
		c2.b *= c.b;
		c2.a *= c.a;
		res += c2;

		//c.r = c.g = c.b = 1 - 0.3f*log(firstZ);
		if ( emissive ) {
			res *= firstShape.mat.emissive;
		}
		return res;
	}
}

Pixel[] generateImage ( ushort width, ushort height ) {

	auto scene = new Scene ();

	Material mat1 = new Material();
	mat1.color = vec4(0.6f,0.2f,0,1);
	mat1.transparency = 0.0f;
	mat1.reflectivity = 0.5f;
	mat1.spec = 1;

	Material mat2 = new Material();
	mat2.color = vec4(0,0.6f,0.1f,1);
	mat2.transparency = 0.0f;
	mat2.reflectivity = 0.5f;
	mat2.spec = 1;	

	Material[] colorMats;
	foreach (i; 0..6) {
		Material mat = new Material ();
		mat.color = hsv2rgb(vec4(i / 5.0f,0.8f,0.7f,1));//vec4(0,0.1f, 0.6f,1);
		mat.transparency = 0.0f;
		mat.reflectivity = 0.2f;
		mat.emissive = 0.5f;
		colorMats ~= mat;
	}

	Shape sh = new Sphere (vec3(0,0,3), 1);
	sh.mat = mat1;
	scene.shapes ~= sh;

	sh = new Sphere (vec3(1.2f,1.2f,3.0f), 1);
	sh.mat = mat2;
	scene.shapes ~= sh;

	sh = new Plane (vec3(-2,0,0), vec3(1,0,0).normalized);
	sh.mat = colorMats[0];
	scene.shapes ~= sh;

	sh = new Plane (vec3(2,0,0), vec3(-1,0,0).normalized);
	sh.mat = colorMats[1];
	scene.shapes ~= sh;

	sh = new Plane (vec3(0,2,0), vec3(0,-1,0).normalized);
	sh.mat = colorMats[2];
	scene.shapes ~= sh;

	sh = new Plane (vec3(0,-2,0), vec3(0,1,0).normalized);
	sh.mat = colorMats[3];
	scene.shapes ~= sh;

	sh = new Plane (vec3(0,0,5), vec3(0,0,-1).normalized);
	sh.mat = colorMats[4];
	scene.shapes ~= sh;

	sh = new Plane (vec3(0,0,-1), vec3(0,0,1).normalized);
	sh.mat = colorMats[5];
	scene.shapes ~= sh;

	scene.lights ~= new Light ( vec3(1f,1.8f,1), vec4(1,1,1,1), 0.5f );
	scene.lights ~= new Light ( vec3(-1f,1.59f,2), vec4(1,1,1,1), 2 );

	auto vp = new Viewport ();
	vp.scene = scene;
	vp.cameraPos = vec3(0,0,0);
	vp.cameraSize = vec2(2f,2f);
	vp.cameraPos.x -= vp.cameraSize.x*0.5f;
	vp.cameraPos.y -= vp.cameraSize.y*0.5f;

	vec4[] pixels = new vec4[width*height];

	vec3[] AAPattern = [
		vec3 ( 0, 0, 0),
		vec3 ( 0.5f, 0.5f, 0),
		vec3 ( -0.5f, 0.5f, 0),
		vec3 ( -0.5f, -0.5f, 0),
		vec3 ( 0.5f, -0.5f, 0),

		vec3 ( 0.5f, 0, 0),
		vec3 ( -0.5f, 0, 0),
		vec3 ( 0, 0.5f, 0),
		vec3 ( 0, -0.5f, 0),
	];

	foreach (ref sample; AAPattern) {
		sample.x *= vp.cameraSize.x / cast(float)width;
		sample.y *= vp.cameraSize.y / cast(float)height;
	}

	writeln("Starting...");

	//auto rend = new DistanceFieldRenderer ();
	auto rend = new Raytracer ();
	rend.AAPattern = AAPattern;
	rend.render (vp, width, height, pixels);
	

	Pixel[] imgPixels = new Pixel[width*height];
	foreach (i, vec4 color; pixels) {
		Pixel px;
		px.r = cast(ubyte)(max(0,min(255,255*color.r)));
		px.g = cast(ubyte)(max(0,min(255,255*color.g)));
		px.b = cast(ubyte)(max(0,min(255,255*color.b)));
		px.a = cast(ubyte)(max(0,min(255,255*color.a)));
		px.a = 255;
		imgPixels[i] = px;
	}
	return imgPixels;
}

class DistanceFieldRenderer : Renderer {
	vec3[] AAPattern;
	Scene scene;

	override void render ( Viewport vp, ushort width, ushort height, vec4[] pixels) {
		auto cameraPos = vp.cameraPos;
		auto cameraSize = vp.cameraSize;
		this.scene = vp.scene;

		foreach ( y; 0 .. height ) {
			writeln(y/(1.0f*height));
			float yf = 1 - (y / cast(float)width);
			foreach ( x; 0 .. width ) {

				float xf = x / cast(float)width;

				vec3 rayOrig = cameraPos + vec3(cameraSize.x*xf, cameraSize.y*yf, 0);
				vec3 rayDir = vec3((xf-0.5f),(yf-0.5f),1).normalized;

				vec4 color = vec4(0,0,0,0);

				float dist;
				vec3 sample = AAPattern[0];
				{
					auto orig = rayOrig + sample;
					auto c = traceField (orig, rayDir);
					color += c;
				}
				
				pixels[x + y*width] = min(color, vec4(1000,1000,1000,1000));
			}
		}
	}

	vec4 traceField ( vec3 orig, vec3 dir ) {
		Shape closestShape = null;
		float totalZ = 0;
		float prevD = 1000000;
		float prevGreatestD = 10000000;
		//write("Start Sample ");
		//write(orig);
		//write(dir);
		//writeln("");
		int steps = 0;

		while(true) {
			float greatestD;
			float d = sampleField (orig, closestShape, greatestD);

			//write(orig);
			//writeln("");

			//write("Total ");
			//write(totalZ);
			//write(" Step: ");
			//writeln(d);

			if ( prevGreatestD < greatestD ) {
				return vec4(0.5,0,0,0);
			}
			prevGreatestD = greatestD;

			if ( d < 0.01 ) {
				return vec4(steps,steps,steps,1)*0.01;
			}
			//writeln(totalZ);
			if ( steps > 1000 ) {
				return vec4(0,0,0,0);
			}

			orig += dir*d*0.5;
			prevD = d;
			totalZ += d;
			steps++;
		}
	}

	float sampleField ( vec3 p, out Shape closestShape, out float greatestD ) {
		float minDist = 10000000;
		closestShape = null;
		greatestD = 0;
		foreach ( shape; scene.shapes ) {
			float d = shape.distance (p);
			if ( d < minDist ) {
				minDist = d;
				closestShape = shape;
			}
			if ( d > greatestD ) {
				greatestD = d;
			}
		}
		return minDist;
	}
}

class Raytracer : Renderer {
	vec3[] AAPattern;

	override void render ( Viewport vp, ushort width, ushort height, vec4[] pixels) {

		auto cameraPos = vp.cameraPos;
		auto cameraSize = vp.cameraSize;
		auto scene = vp.scene;

		foreach ( y; 0 .. height ) {
			writeln(y/(1.0f*height));
			float yf = 1 - (y / cast(float)width);
			foreach ( x; 0 .. width ) {

				float xf = x / cast(float)width;

				vec3 rayOrig = cameraPos + vec3(cameraSize.x*xf, cameraSize.y*yf, 0);
				vec3 rayDir = vec3((xf-0.5f),(yf-0.5f),1).normalized;

				vec4 color = vec4(0,0,0,0);

				float dist;
				vec3 sample = AAPattern[0];
				{
					vec4 c = scene.getColor ( rayOrig + sample, rayDir, 0, dist);
					color += c;
				}
				
				pixels[x + y*width] = min(color, vec4(1000,1000,1000,1000));
			}
		}

		int[] dx = [ -1, 0,0, 1, 1];
		int[] dy = [ 0, -1, 1, 0, 1];

		writeln("Supersampling...");

		

		foreach ( y; 0 .. height ) {
			writeln(y/(1.0f*height));
			float yf = 1 - (y / cast(float)width);
			foreach ( x; 0 .. width ) {

				float xf = x / cast(float)width;

				vec4 color = pixels[x + y*width];
				vec4 mn = color;
				vec4 mx = color;
				for ( int i = 2; i < dx.length; i++ ) {
					int nx = x + dx[i];
					int ny = y + dy[i];
					if ( nx >= 0 && ny >= 0 && nx < width && ny < height ) {
						//writeln(pixels[nx + ny*width]);
						mn = min(mn,pixels[nx + ny*width]);
						mx = max(mx,pixels[nx + ny*width]);
					}
				}

				vec3 rayOrig = cameraPos + vec3(cameraSize.x*xf, cameraSize.y*yf, 0);
				vec3 rayDir = vec3((xf-0.5f),(yf-0.5f),1).normalized;

				

				ulong mxi = 1 + cast(ulong)(AAPattern.length * (mx-mn).magnitude_squared*(1.0f/(0.4f*0.4f)));
				mxi = min(mxi, AAPattern.length);
				//writeln(mxi);
				//writeln(color);

				float dist;
				foreach (i; 1..mxi) {
					vec4 c = scene.getColor ( rayOrig + AAPattern[i], rayDir, 0, dist);
					color += c;
				}
				color *= (1.0f/mxi);
				//color = vec4(mxi/10.0f, 0,0, 1);
				//writeln((mx-mn).magnitude_squared);
				//color = vec4(0.01f*(mx-mn).magnitude_squared*(1.0f/(0.04f*0.04f)), 0,0,1);
				//writeln(color);

				pixels[x + y*width] = color;
			}
		}
	}
}