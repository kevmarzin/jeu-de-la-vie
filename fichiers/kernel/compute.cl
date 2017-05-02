static unsigned getLinePosition(unsigned x, unsigned y){
	return x * DIM + y;
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
	unsigned DEAD_STATE = 0x0;
	unsigned ALIVE_STATE = 0xffff00ff;
	
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

	bool next_color_alive = 
		((in [getLinePosition(x,y)] == DEAD_STATE  && cpt_alive_cell == 3)  || 
         (in [getLinePosition(x,y)] == ALIVE_STATE && (cpt_alive_cell == 2 || cpt_alive_cell == 3)));

  	out [getLinePosition(x,y)] = next_color_alive ? ALIVE_STATE : next_color_alive;
}

__kernel void compute_cl_optimized (__global unsigned *in, __global unsigned *out)
{
  	__local unsigned tile [TILEX+2][TILEY+2];
	int x = get_global_id (0);
	int y = get_global_id (1);

	int xLoc = get_local_id (0) + 1;
	int yLoc = get_local_id (1) + 1;

	int cpt_alive_cell = 0;
	unsigned DEAD_STATE = 0x0;
	unsigned ALIVE_STATE = 0xffff00ff;

	tile [xLoc][yLoc] = in [getLinePosition(x, y)];

	// coin haut gauche
	if (xLoc == 1 && yLoc == 1){
		tile [0][0] = (x == 0 || y == 0) ? 0x0 : in [getLinePosition(x-1, y-1)];
	}
	// coin haut droit
	else if (xLoc == 1 && yLoc == TILEY){
		tile [0][(TILEY+1)] = (x == 0 || y == (DIM-1)) ? 0x0 : in [getLinePosition(x-1, y+1)];
	}
	// coin bas gauche
	else if (xLoc == TILEX && yLoc == 1){
		tile [(TILEX+1)][0] = (x == (DIM-1) || y == 0) ? 0x0 : in [getLinePosition(x+1, y-1)];
	}
	// coin bas droit
	else if (xLoc == TILEX && yLoc == TILEY){
		tile [(TILEX+1)][(TILEY+1)] = (x == (DIM-1) || y == (DIM-1)) ? 0x0 : in [getLinePosition(x+1, y+1)];
	}
	else if (xLoc == 1) {
		tile [0][yLoc] = (x == 0) ? 0x0 : in [getLinePosition(x-1, y)];
	}
	// bordure gauche
	else if (xLoc == 1) {
		tile [xLoc][0] = (y == 0) ? 0x0 : in [getLinePosition(x, y-1)];
	}
	else if (xLoc == TILEX) {
		tile [(TILEX+1)][yLoc] = (x == DIM-1) ? 0x0 : in [getLinePosition(x+1, y)];
	}
	else if (yLoc == TILEY) {
		tile [xLoc][(TILEX+1)] = (y == DIM-1) ? 0x0 : in [getLinePosition(x, y+1)];
	}

  	barrier (CLK_LOCAL_MEM_FENCE);

  	cpt_alive_cell += (tile[xLoc-1][yLoc-1] != DEAD_STATE); // pixel haut dessus à gauche
	cpt_alive_cell += (tile[xLoc-1][yLoc] != DEAD_STATE); // pixel haut dessus
	cpt_alive_cell += (tile[xLoc-1][yLoc+1] != DEAD_STATE); // pixel haut dessus à droite
	cpt_alive_cell += (tile[xLoc][yLoc+1] != DEAD_STATE); // pixel à droite
	cpt_alive_cell += (tile[xLoc+1][yLoc+1] != DEAD_STATE); // pixel en bas à droite
	cpt_alive_cell += (tile[xLoc+1][yLoc] != DEAD_STATE); // pixel en bas
	cpt_alive_cell += (tile[xLoc+1][yLoc-1] != DEAD_STATE); // pixel en bas à gauche
	cpt_alive_cell += (tile[xLoc][yLoc-1] != DEAD_STATE); // pixel en bas à gauche

	bool next_color_alive = 
		((tile[xLoc][yLoc] == DEAD_STATE && cpt_alive_cell == 3) || 
         (tile[xLoc][yLoc] != DEAD_STATE && (cpt_alive_cell == 2 || cpt_alive_cell == 3)));

  	out [getLinePosition(x,y)] = next_color_alive ? ALIVE_STATE : next_color_alive;
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
