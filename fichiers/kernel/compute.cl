static unsigned getLinePosition(unsigned l, unsigned c)
{
	return l * DIM + c;
}

// NE PAS MODIFIER
static int4 color_to_int4 (unsigned c)
{
  uchar4 ci = *(uchar4 *) &c;
  return convert_int4 (ci);
}

// NE PAS MODIFIER
static unsigned int4_to_color (int4 i)
{
  return (unsigned) convert_uchar4 (i);
}

__kernel void transpose_naif (__global unsigned *in, __global unsigned *out)
{
  int x = get_global_id (0);
  int y = get_global_id (1);

  out [x * DIM + y] = in [y * DIM + x];
}

__kernel void transpose (__global unsigned *in, __global unsigned *out)
{
  __local unsigned tile [TILEX][TILEY+1];
  int x = get_global_id (0);
  int y = get_global_id (1);
  int xloc = get_local_id (0);
  int yloc = get_local_id (1);

  tile [xloc][yloc] = in [y * DIM + x];

  barrier (CLK_LOCAL_MEM_FENCE);

  out [(x - xloc + yloc) * DIM + y - yloc + xloc] = tile [yloc][xloc];
}

__kernel void compute_cl_naif (__global unsigned *in, __global unsigned *out){

  int x = get_global_id (0);
  int y = get_global_id (1);

	int cpt_alive_cell = 0;
	unsigned DEAD_STATE = int4_to_color((int4) 0);
	unsigned ALIVE_STATE = int4_to_color((int4) (0xff, 0xff, 0x0, 0xff));
	
	// top
	if (x > 0) cpt_alive_cell += (in[getLinePosition (x-1, y)] != DEAD_STATE);
	
	// top-right corner
	if (x > 0 && y < DIM-1) cpt_alive_cell += (in[getLinePosition(x-1, y+1)] != DEAD_STATE);
	
	// right
	if (y < DIM-1) cpt_alive_cell += (in[getLinePosition (x, y+1)] != DEAD_STATE);
	
	// right-bottom corner
	if (x < DIM-1 && y < DIM-1)	cpt_alive_cell += (in[getLinePosition (x+1, y+1)] != DEAD_STATE);
	
	// bottom
	if (x < DIM-1) cpt_alive_cell += (in[getLinePosition (x+1, y)] != DEAD_STATE);
	
	// left-bottom corner
	if (x < DIM-1 && y > 0)	cpt_alive_cell += (in[getLinePosition (x+1, y-1)] != DEAD_STATE);
	
	// left
	if (y > 0) cpt_alive_cell += (in[getLinePosition (x, y-1)] != DEAD_STATE);
	
	// top-left corner
	if (x > 0 && y > 0)	cpt_alive_cell += (in[getLinePosition(x-1, y-1)] != DEAD_STATE);

	unsigned next_color = 
	((in [getLinePosition(x,y)] == DEAD_STATE  && cpt_alive_cell == 3)  || 
         (in [getLinePosition(x,y)] == ALIVE_STATE && (cpt_alive_cell == 2 || cpt_alive_cell == 3)));

  	out [getLinePosition(y,x)] = next_color;
}


// NE PAS MODIFIER
static float4 color_scatter (unsigned c)
{
  uchar4 ci;

  ci.s0123 = (*((uchar4 *) &c)).s3210;
  return convert_float4 (ci) / (float4) 255;
}

// NE PAS MODIFIER: ce noyau est appelé lorsqu'une mise à jour de la
// texture de l'image affichée est requise
__kernel void update_texture (__global unsigned *cur, __write_only image2d_t tex)
{
  int y = get_global_id (1);
  int x = get_global_id (0);
  int2 pos = (int2)(x, y);
  unsigned c;

  c = cur [y * DIM + x];

  write_imagef (tex, pos, color_scatter (c));
}
