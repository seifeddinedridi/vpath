#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <gl/glut.h>
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>

#define NB_THREADS_X 16
///////////////////////////////////////////////////////////////////
/////////////// GLOBAL VARIABLES //////////////////////////////////
///////////////////////////////////////////////////////////////////
const int width = 600, height = 600;
unsigned int *seeds, samplesPerPixel = 0;
struct Vec *image, *d_image;
struct Sphere *spheres, *d_spheres, *d_homogeneousMedium;
int nbSpheres;
curandStateXORWOW_t *d_states;
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////

#define CHECK_CUDA_ERRORS(call) {\
	cudaError err = call;\
	if( err != cudaSuccess) {\
	fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",\
	__FILE__, __LINE__, cudaGetErrorString( err) );\
	exit(EXIT_FAILURE);\
	} }
	
__global__ void initCurandStates(unsigned int *d_seeds, curandStateXORWOW_t *d_states, int width, int height) {
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	if (ix >= width || iy >= height)
		return;
	int idx = ix + iy * width;
	curand_init(d_seeds[idx], idx*0, 0, &d_states[idx]);
}

inline float clamp(float x) { 
	return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x; 
}

inline int toInt(float x) { 
	return int(powf(clamp(x), 1 / 2.2f) * 255.0f + 0.5f); 
}

namespace XORShift { // XOR shift PRNG
	unsigned int x = 123456789;
	unsigned int y = 362436069;
	unsigned int z = 521288629;
	unsigned int w = 88675123; 
	inline unsigned int frand() { 
		unsigned int t;
		t = x ^ (x << 11);
		x = y; y = z; z = w;
		return (w = (w ^ (w >> 19)) ^ (t ^ (t >> 8))); 
	}
}

struct Vec {        
	float x, y, z;                  // position, also color (r,g,b)
	__host__ __device__ Vec(float x_= 0, float y_= 0, float z_= 0) { 
		x = x_; 
		y = y_; 
		z = z_; 
	}
	inline __host__ __device__ Vec operator+(const Vec &b) const { 
		return Vec(x+b.x, y+b.y, z+b.z); 
	}
	inline __host__ __device__ Vec operator-(const Vec &b) const { 
		return Vec(x-b.x, y-b.y, z-b.z); 
	}
	inline __host__ __device__ Vec operator*(float b) const { 
		return Vec(x*b, y*b, z*b); 
	}
	inline __host__ __device__ Vec mult(const Vec &b) const { 
		return Vec(x*b.x, y*b.y, z*b.z); 
	}
	inline __host__ __device__ Vec& norm() { 
		return *this = *this * (1 / sqrtf(x*x + y*y + z*z)); 
	}
	inline __host__ __device__ float length() {
		return sqrtf(x*x + y*y + z*z); 
	}
	inline __host__ __device__ float dot(const Vec &b) const { 
		return x*b.x + y*b.y + z*b.z; 
	} 
	inline __host__ __device__ Vec operator%(const Vec &b) const {
		return Vec(y*b.z - z*b.y, z*b.x - x*b.z, x*b.y - y*b.x);
	}
};

struct Ray { 
	Vec o, d; 
	inline __host__ __device__ Ray() {} 
	inline __host__ __device__ Ray(Vec o_, Vec d_) : o(o_), d(d_) {} 
};

enum Refl_t { DIFF = 1 << 0, SPEC = 1 << 1, REFR = 1 << 2, VOL = 1 << 3 };  // Material types

struct VolumetricProps {
	__device__ VolumetricProps() {}
	__device__ VolumetricProps(Vec c_s, Vec c_a, float sig_s, float sig_a):	scatteringColor(c_s), absorptionColor(c_a), 
																			sigma_s(sig_s), sigma_a(sig_a), sigma_t(sig_s + sig_a) {}

	float sigma_s, sigma_a, sigma_t; // scattering, absorption and extinction coefficients
	Vec scatteringColor, absorptionColor; // scattering, absorption
};

struct Sphere {
	float rad; // radius
	Vec p, e, c; // position, emission, color
	Refl_t refl; // reflection type (DIFFuse, SPECular, REFRactive)
	VolumetricProps volProps;
	float ior;
	__host__ __device__ Sphere() {}
	__host__ __device__ Sphere(float rad_, Vec p_, Vec e_, Vec c_, Refl_t refl_, VolumetricProps vol_p = VolumetricProps(), float _ior = 1.5f):	rad(rad_), p(p_), e(e_), c(c_), 
																																refl(refl_),  volProps(vol_p), ior(_ior) {}
	__host__ __device__  float intersect(const Ray *r, float *tin, float *tout) const {
		Vec op = p - r->o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
		float t, eps = 1e-3f, b = op.dot(r->d), det = b * b - op.dot(op) + rad*rad;
		if (det < 0.0f) return 0.0f; 
		else 
			det = sqrtf(det);
		if (tin && tout) {
			*tin	= (b - det <= 0.0f) ? 0.0f : b - det;
			*tout 	= b + det;
		}
		return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0.0f);
	}
};

__device__ bool intersect(const Ray *r, const Sphere *d_spheres, int nbSpheres, float &t, int &id) {
	float d, inf = t = 1e7f, tnear, tfar;
	for(int i = nbSpheres; i--;) 
		if((d = d_spheres[i].intersect(r, &tnear, &tfar)) && (d < t)) {
			t=d;
			id=i;
		}
	return t < inf;
}

__device__ inline float sampleSegment(float epsilon, float sigma) {
	return -logf(1.0f - epsilon) / sigma;
}
__device__ inline Vec sampleSphere(float e1, float e2) {
	float z = 1.0f - 2.0f * e1, sint = sqrtf(1.0f - z * z);
	return Vec(cosf(2.0f * M_PI * e2) * sint, sinf(2.0f * M_PI * e2) * sint, z);
}
__device__ inline Vec sampleHG(float g, float e1, float e2) {
	//float s=2.0*e1-1.0, f = (1.0-g*g)/(1.0+g*s), cost = 0.5*(1.0/g)*(1.0+g*g-f*f), sint = sqrtf(1.0-cost*cost);
	float s = 1.0f - 2.0f*e1, denom = 1.0f + g*s;
	float cost = (s + 2.0f*g*g*g*(-1.0f + e1) * e1 + g*g*s + 2.0f*g*(1.0f - e1 + e1*e1)) / (denom * denom), sint = sqrtf(1.0f - cost*cost);
	return Vec(cosf(2.0f * M_PI * e2) * sint, sinf(2.0f * M_PI * e2) * sint, cost);
}
__device__ inline void generateOrthoBasis(Vec &u, Vec &v, Vec w) {
	Vec coVec;
	if (fabs(w.x) <= fabs(w.y))
		if (fabs(w.x) <= fabs(w.z)) coVec = Vec(0,-w.z,w.y);
		else coVec = Vec(-w.y,w.x,0);
	else if (fabs(w.y) <= fabs(w.z)) coVec = Vec(-w.z,0,w.x);
	else coVec = Vec(-w.y,w.x,0);
	coVec.norm();
	u = w%coVec,
	v = w%u;
}
__device__ inline float scatter(const Ray &r, Ray &sRay, float sigma_s, float &s, float e0, float e1, float e2) {
	s = sampleSegment(e0, sigma_s);
	Vec x = r.o + r.d * s;
	//Vec dir = sampleSphere(e1, e2); //Sample a direction ~ uniform phase function
	Vec dir = sampleHG(-0.0f, e1, e2); //Sample a direction ~ Henyey-Greenstein's phase function
	Vec u,v;
	generateOrthoBasis(u, v, r.d);
	dir = u * dir.x + v * dir.y + r.d * dir.z;
	sRay = Ray(x, dir);
	return 1.0f;
}

__global__ void rendering_kernel(Vec *d_image, int width, int height, Sphere *d_spheres, int nbSpheres, Sphere *d_homogeneousMedium, curandStateXORWOW_t *d_states) {
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	if (ix >= width || iy >= height)
		return;

	int idx = ix + width * iy;
	curandStateXORWOW_t state = d_states[idx];

	// Make a primary ray
	Ray cam(Vec(0, 4.0f, -15.0f), Vec(0, -0.1f, -1.0f).norm());
	float aspectRatio = (float) width / (float) height;
	Vec cx = Vec(aspectRatio), cy = (cx % cam.d).norm();
	float rasterX = 1.0f * (ix + curand_uniform(&state)) / (float)width - 0.5f;
	float rasterY = -1.0f * (iy + curand_uniform(&state)) / (float)height + 0.5f;
	Vec d = cx * rasterX + cy * rasterY + cam.d;
	Ray r(cam.o, d.norm());
	Vec pathThroughput;
	Vec brdfsProduct = Vec(1.0f, 1.0f, 1.0f);
	for (int depth = 0; depth < 200; ++depth) {
		int id = 0;
		float t_s, t_m = 1e7f, tnear_m, tfar_m;
		Vec absorption(1.0f, 1.0f, 1.0f);
		bool intrsctmd = (t_m = d_homogeneousMedium->intersect(&r, &tnear_m, &tfar_m)) > 0.0f;
		bool intrscts = intersect(&r, d_spheres, nbSpheres, t_s, id);
		if (!intrscts && !intrsctmd)
			break;
		bool doAtmosphericScattering = (intrsctmd && (!intrscts || (t_m <= t_s || (t_s >= tnear_m && t_s <= tfar_m))));
		if (intrscts && (d_spheres[id].refl == REFR || (d_spheres[id].refl & VOL) == VOL) && (r.o + r.d * t_s - d_spheres[id].p).dot(r.d) >= 0.0f)
			doAtmosphericScattering = false;
		Sphere *obj = &d_spheres[id];
		float t = t_s;
		if (doAtmosphericScattering) {
			obj = d_homogeneousMedium;
			t = t_m;
		}
		if ((obj->refl & VOL) == VOL && (r.o + r.d * t - obj->p).dot(r.d) >= 0.0f) {
			Ray sRay;
			float e0 = curand_uniform(&state), e1 = curand_uniform(&state), e2 = curand_uniform(&state);
			const VolumetricProps &volProps = obj->volProps;
			float s, ms = /*(volProps.sigma_s / volProps.sigma_t) */ scatter(r, sRay, volProps.sigma_s, s, e0, e1, e2);
			float distToExit = t_s < t ? t_s : t;
			if (s <= distToExit && volProps.sigma_s > 0) { 
				r = sRay;
				brdfsProduct = brdfsProduct.mult(volProps.scatteringColor * ms);
				absorption = Vec(1.0f, 1.0f, 1.0f) + volProps.absorptionColor * (expf(-volProps.sigma_a * s) - 1.0f);
				brdfsProduct = brdfsProduct.mult(absorption);
				continue;
			}
			float dist = t_m;
			// Ray is probably leaving the medium
			if (intrscts && t_s <= t) {
				obj = &d_spheres[id];
				t = t_s;
				dist = t_s;
			}
			absorption = Vec(1.0f, 1.0f, 1.0f) + volProps.absorptionColor * (expf(-volProps.sigma_t * dist) - 1.0f);
		}
		Vec d, x = r.o + r.d * t, n = (x - obj->p).norm(), nl = n.dot(r.d) < 0 ? n : n * -1.0f, f = obj->c.mult(absorption), Le = obj->e.mult(absorption);
		if ((obj->refl & DIFF) == DIFF) {		// Ideal DIFFUSE reflection
			float rnx = curand_uniform(&state);
			float rny = curand_uniform(&state);
	
			float r1 = 2.0f * M_PI * rnx, r2 = rny, r2s = sqrtf(r2);
			Vec w = nl, u = ((fabsf(w.x) > 0.1f ? Vec(0.0f, 1.0f) : Vec(1.0f)) % w).norm(), v = w % u;
			d = Vec(u * cosf(r1) * r2s + v * sinf(r1) * r2s + w * sqrtf(1.0f - r2)).norm();
		} 
		else if ((obj->refl & SPEC) == SPEC)		// Ideal SPECULAR reflection
			d = (r.d - n * 2.0f * n.dot(r.d));
		else if ((obj->refl & REFR) == REFR) {	// Ideal dielectric REFRACTION
			Vec reflDir(r.d - n * 2.0f * n.dot(r.d));
			bool into = n.dot(nl) > 0.0f;  // Ray from outside going in?
			float nc = 1.0f, nt = obj->ior, nnt = into ? nc / nt : nt / nc, ddn = r.d.dot(nl), cos2t;
			if ((cos2t = 1.0f - nnt * nnt * (1.0f - ddn * ddn)) < 0.0f) // Total internal reflection
				d = reflDir;
			else {
				Vec tdir = (r.d * nnt - n * ((into ? 1.0f : -1.0f) * (ddn * nnt + sqrt(cos2t)))).norm();
				float a = nt - nc, b = nt + nc, R0 = a*a / (b*b), c = 1.0f - (into ? -ddn : tdir.dot(n));
				float Re = R0 + (1.0f - R0)*c*c*c*c*c, Tr = 1.0f - Re, P = 0.25f + 0.5f * Re, RP = Re / P, TP = Tr / (1.0f - P);
				if (curand_uniform(&state) < P) {
					f = absorption.mult(Vec(1.0f, 1.0f, 1.0f) * RP);
					d = reflDir;
				}
				else {
					f = f * TP;
					d = tdir;
				}
			}
		}
		pathThroughput = pathThroughput + brdfsProduct.mult(Le);
		brdfsProduct = brdfsProduct.mult(f);
		r = Ray(x, d);
		if (Le.x != 0.0f || Le.y != 0.0f || Le.z != 0.0f) // Break if a ray hits a light source
			break;
	}
	d_states[idx] = state;
	d_image[idx] = pathThroughput;
}

void render(Vec *image, int width, int height, Sphere *spheres, int nbSpheres, Sphere *d_homogeneousMedium) {
	int nbbx = (width + NB_THREADS_X - 1) / NB_THREADS_X;
	int nbby = (height + NB_THREADS_X - 1) / NB_THREADS_X;
	dim3 nbBlocks(nbbx,nbby);
	//printf("number of blocks per dimension %d\n", nbb);
	dim3 threadsPerBlock(NB_THREADS_X, NB_THREADS_X);
	cudaMemset(d_image, 0, sizeof(Vec) * width * height);
	rendering_kernel<<<nbBlocks, threadsPerBlock>>>(d_image, width, height, d_spheres, nbSpheres, d_homogeneousMedium, d_states);

	Vec *h_tmp_image = new Vec[width * height];
	// Copy rendered image back to host memory
	CHECK_CUDA_ERRORS(cudaMemcpy(h_tmp_image, d_image, sizeof(Vec) * width * height, cudaMemcpyDeviceToHost));

	for (int i = 0; i < width * height; ++i)
		image[i] = image[i] + h_tmp_image[i];

	delete[] h_tmp_image;
}

void initEngine() {
	// Create seed array
	seeds = new unsigned int[width * height];
	// Create buffer image
	image = new Vec[width * height];
	Sphere tmpSpheres[] = {//Scene: radius, position, emission, color, material 
		Sphere(0.5f, Vec(0.0f, 4.0f, -24.0f), Vec(2.0f, 2.0f, 2.0f) * 10.0f, Vec(), DIFF), // Light source 
		Sphere(2.0f, Vec(2.0f, 1.0f, -25.0f), Vec(), Vec(1, 1, 1) * 0.8f, Refl_t(REFR | VOL), VolumetricProps(Vec(1.0, 1.0, 1.0), Vec(0.0f, 0.2f, 0.9f), 4.0f, 3.1f), 1.33f),
		Sphere(2.0f, Vec(-2.0f, 1.0f, -25.0f), Vec(), Vec(1, 1, 1) * 0.8f, Refl_t(REFR | VOL), VolumetricProps(Vec(1.0, 1.0, 1.0), Vec(0.9f, 0.2f, 0.9f), 4.0f, 0.1f), 1.33f),
		Sphere(1000.0f, Vec(0.0f, -1001.0f, -25.0f), Vec(), Vec(1, 1, 1) * 0.8f, DIFF),
	};

	// Define scene's medium
	Sphere medium(20.0f, Vec(0.0f, 0.0f, -25000.0f), Vec(), Vec(), VOL, VolumetricProps(Vec(0.8f, 0.8f, 0.8f), Vec(1.0f, 1.0f, 1.0f), 0.08f, 0.001f));

	CHECK_CUDA_ERRORS(cudaMalloc(&d_homogeneousMedium, sizeof(Sphere)));
	CHECK_CUDA_ERRORS(cudaMemcpy(d_homogeneousMedium, &medium, sizeof(Sphere), cudaMemcpyHostToDevice));

	nbSpheres = sizeof(tmpSpheres) / sizeof(Sphere);
	spheres = new Sphere[nbSpheres];
	memcpy(spheres, tmpSpheres, sizeof(tmpSpheres));
	unsigned int *d_seeds;
	// Initialize gpu stuff
	cudaMalloc(&d_seeds, sizeof(unsigned int) * width * height);
	CHECK_CUDA_ERRORS(cudaMalloc(&d_image, sizeof(Vec) * width * height));
	
	int nbbx = (width + NB_THREADS_X - 1) / NB_THREADS_X;
	int nbby = (height + NB_THREADS_X - 1) / NB_THREADS_X;
	dim3 nbBlocks(nbbx, nbby);
	dim3 threadsPerBlock(NB_THREADS_X, NB_THREADS_X);
	
	for (int i = 0; i < width * height; ++i)
		seeds[i] = XORShift::frand();

	cudaMemcpy(d_seeds, seeds, sizeof(unsigned int) * width * height, cudaMemcpyHostToDevice);

	cudaMalloc(&d_states, sizeof(curandStateXORWOW_t) * nbBlocks.x * nbBlocks.y * NB_THREADS_X * NB_THREADS_X);
	initCurandStates<<<nbBlocks, threadsPerBlock>>>(d_seeds, d_states, width, height);

	CHECK_CUDA_ERRORS(cudaFree(d_seeds));
	delete[] seeds;

	// Allocate and copy scene geometry to the device
	CHECK_CUDA_ERRORS(cudaMalloc((void**)&d_spheres, sizeof(Sphere) * nbSpheres));
	CHECK_CUDA_ERRORS(cudaMemcpy(d_spheres, spheres, sizeof(Sphere) * nbSpheres, cudaMemcpyHostToDevice));

	// Setup viewport
	glViewport( 0, 0, width, height);

	// Set up an orthographic view
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0.0f, 1.0f, 0.0f, 1.0f ,-1.0f, 1.0f);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	// Create a texture for displaying the render:
	glBindTexture(GL_TEXTURE_2D, 1);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	// Use nearest-neighbor point sampling instead of linear interpolation:
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); //glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); //glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
}

void shutdownEngine() {
	// Free any allocated memory
	delete[] spheres;
	delete[] image;
	CHECK_CUDA_ERRORS(cudaFree(d_spheres));
	CHECK_CUDA_ERRORS(cudaFree(d_homogeneousMedium));
	CHECK_CUDA_ERRORS(cudaFree(d_image));
	CHECK_CUDA_ERRORS(cudaFree(d_states));
}

uchar3 floatTo8Bit(const Vec & color) {
	uchar3 eightBitColor;
	eightBitColor.x = toInt(color.x);
	eightBitColor.y = toInt(color.y);
	eightBitColor.z = toInt(color.z);
	return eightBitColor;
}

void printStringToRaster(char *info, float x, float y) {
	glRasterPos2f(x, y); 
	for (unsigned int i = 0; i < strlen(info); i++) {
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, info[i]);
	}
}

void render() {
	static float totalElapsedTime = 0;
	// Measuring rendering time (in millisecond)
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);

	const int spp = 1;
	for (int s = 0; s < spp; ++s) {
		render(image, width, height, spheres, nbSpheres, d_homogeneousMedium);
	}

	samplesPerPixel += spp;

	cudaThreadSynchronize();
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, end);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	totalElapsedTime += elapsedTime;
	
	glClear(GL_COLOR_BUFFER_BIT);

	uchar3* imageData = new uchar3[width * height];
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			Vec c = image[i + width * j];
			imageData[i + width * j] = floatTo8Bit(c * (1.0f / (float)(samplesPerPixel)));
		}
	}
	glTexImage2D (GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, imageData);
	delete [] imageData; // glTexImage2D makes a copy of the data, so the original data can (and should!) be deleted here (otherwise it will leak memory like a madman).

	glEnable(GL_TEXTURE_2D);
	// Show the texture:
	glBindTexture (GL_TEXTURE_2D, 1);
	glBegin (GL_QUADS);
	glTexCoord2f (0.0, 0.0);
	glVertex3f (0.0, 1.0, 0.0);
	glTexCoord2f (1.0, 0.0);
	glVertex3f (1.0, 1.0, 0.0);
	glTexCoord2f (1.0, 1.0);
	glVertex3f (1.0, 0.0, 0.0);
	glTexCoord2f (0.0, 1.0);
	glVertex3f (0.0, 0.0, 0.0);
	glEnd ();
	glDisable(GL_TEXTURE_2D);
	
	char info[1024];
	sprintf(info, "Iteration: %u", samplesPerPixel);
	printStringToRaster(info, 0.01f, 0.972f);
	
	sprintf(info, "Samples / s: %.2fk", samplesPerPixel * width * height / totalElapsedTime);
	printStringToRaster(info, 0.01f, 0.942f);
	
	sprintf(info, "Elapsed time: %.2fs", totalElapsedTime * 0.001f);
	printStringToRaster(info, 0.01f, 0.912f); 

	glutSwapBuffers();
	glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y) {
	if (key == 's') {
		char filename[1024];
		sprintf(filename, "image%d.ppm", samplesPerPixel);
		FILE *f = fopen(filename, "w"); // Write image to PPM file.
		fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
		for (int i = 0; i < width * height; i++)
			fprintf(f,"%d %d %d ", toInt(image[i].x / samplesPerPixel), toInt(image[i].y / samplesPerPixel), toInt(image[i].z / samplesPerPixel));
		}
}

int main(int argc, char **argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(width, height);
	glutCreateWindow("Small Volumetric GPU Path Tracer written by Seifeddine Dridi");
	// Setup callback functions
	glutDisplayFunc(render);
	glutIdleFunc(render);
	glutKeyboardFunc(keyboard);

	initEngine();
	glutMainLoop();
	shutdownEngine();
	return 0;
}

