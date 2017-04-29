
#include "compute.h"
#include "graphics.h"
#include "debug.h"
#include "ocl.h"

#include <stdbool.h>

unsigned version = 0;

void first_touch_v1 (void);
void first_touch_v2 (void);

unsigned compute_seq_base (unsigned nb_iter);
unsigned compute_seq_tile (unsigned nb_iter);
unsigned compute_seq_tile_optimized (unsigned nb_iter);
unsigned compute_omp_for_base (unsigned nb_iter);
unsigned compute_omp_for_tile (unsigned nb_iter);
unsigned compute_v3 (unsigned nb_iter);

void_func_t first_touch [] = {
  NULL,
  NULL,
  NULL,
  first_touch_v1,
  first_touch_v2,
  NULL,
};

int_func_t compute [] = {
  compute_seq_base,
  compute_seq_tile,
  compute_seq_tile_optimized,
  compute_omp_for_base,
  compute_omp_for_tile,
  compute_v3,
};

char *version_name [] = {
  "Séquentielle de base",
  "Séquentielle tuillée",
  "Séquentielle tuillée optimisée",
  "OpenMP (for) de base",
  "OpenMP (for) tuillée",
  "OpenCL",
};

unsigned opencl_used [] = {
  0,
  0,
  0,
  0,
  0,
  1,
};

int **tile_calc = NULL;

///////////////////////////// Fonctions

// Calcul le nombre de cellule vivante autour d'une cellule gràce à ses coordonnées
int number_arround_alive_cell (int i, int j){
	int cpt_alive_cell = 0;
	
	// top
	if (i > 0) cpt_alive_cell += (cur_img (i-1, j) != DEAD_STATE);
	
	// top-right corner
	if (i > 0 && j < DIM-1) cpt_alive_cell += (cur_img (i-1, j+1) != DEAD_STATE);
	
	// right
	if (j < DIM-1) cpt_alive_cell += (cur_img (i, j+1) != DEAD_STATE);
	
	// right-bottom corner
	if (i < DIM-1 && j < DIM-1)	cpt_alive_cell += (cur_img (i+1, j+1) != DEAD_STATE);
	
	// bottom
	if (i < DIM-1) cpt_alive_cell += (cur_img (i+1, j) != DEAD_STATE);
	
	// left-bottom corner
	if (i < DIM-1 && j > 0)	cpt_alive_cell += (cur_img (i+1, j-1) != DEAD_STATE);
	
	// left
	if (j > 0) cpt_alive_cell += (cur_img (i, j-1) != DEAD_STATE);
	
	// top-left corner
	if (i > 0 && j > 0)	cpt_alive_cell += (cur_img (i-1, j-1) != DEAD_STATE);
	
	return cpt_alive_cell;
}

// Calcul le prochain état du pixel ligne l et colonne c
// Retourne vrai si le pixel (l, c) a changé, faux sinon.
bool set_next_state (int l, int c) {
    
    int cell_arround = number_arround_alive_cell (l, c);
    
    Uint32 current_state = cur_img (l, c);
    Uint32 next_state;
			
    bool alive_on_next_step =
        ((current_state == DEAD_STATE  && cell_arround == 3)  || // cell is dead and 3 cells alive arround
         (current_state == ALIVE_STATE && (cell_arround == 2 || cell_arround == 3))); // cell is alive and 2 or 3 cells alive arround

    next_img (l, c) = (next_state = (alive_on_next_step ? ALIVE_STATE : DEAD_STATE));
    
    return next_state != current_state;
}

void update_tile_calc (){
	int local_l, local_c;
	int l, c;
	int cell_arround;
	Uint32 current_state;
	
	int nb_tile = DIM / TILE_SIZE;
	
	// parcours de toute les tuiles
	for (int l_tile = 0; l_tile < nb_tile; l_tile++){
		for (int c_tile = 0; c_tile < nb_tile; c_tile++){
			// Bord haut
			if (!tile_calc[l_tile][c_tile] && l_tile > 0){
				local_l = 0;
				l = l_tile * TILE_SIZE + local_l;
				for (local_c = 0; local_c < TILE_SIZE; local_c++){
					c = c_tile * TILE_SIZE + local_c;
					
					current_state = cur_img (l, c);
					cell_arround = number_arround_alive_cell (l, c);
					
					if ((current_state == DEAD_STATE  && cell_arround == 3) || // cell is dead and 3 cells alive arround
						(current_state == ALIVE_STATE && (cell_arround == 2 || cell_arround == 3))){ // cell is alive and 2 or 3 cells alive arround
						tile_calc[l_tile][c_tile] = true;
						break;
					}
				}
			}
			
			// Board gauche
			if (!tile_calc[l_tile][c_tile] && c_tile > 0){ 
				local_c = 0;
				c = c_tile * TILE_SIZE + local_c;
				for (local_l = 0; local_l < TILE_SIZE; local_l++){
					l = l_tile * TILE_SIZE + local_l;
					
					current_state = cur_img (l, c);
					cell_arround = number_arround_alive_cell (l, c);
					
					if ((current_state == DEAD_STATE  && cell_arround == 3) || // cell is dead and 3 cells alive arround
						(current_state == ALIVE_STATE && (cell_arround == 2 || cell_arround == 3))){ // cell is alive and 2 or 3 cells alive arround
						tile_calc[l_tile][c_tile] = true;
						break;
					}
				}
			}
				
			// Board droit
			if (!tile_calc[l_tile][c_tile] && c_tile < nb_tile - 1){ 
				local_c = TILE_SIZE - 1;
				c = c_tile * TILE_SIZE + local_c;
				for (local_l = 0; local_l < TILE_SIZE; local_l++){
					l = l_tile * TILE_SIZE + local_l;
					
					current_state = cur_img (l, c);
					cell_arround = number_arround_alive_cell (l, c);
					
					if ((current_state == DEAD_STATE  && cell_arround == 3) || // cell is dead and 3 cells alive arround
						(current_state == ALIVE_STATE && (cell_arround == 2 || cell_arround == 3))){ // cell is alive and 2 or 3 cells alive arround
						tile_calc[l_tile][c_tile] = true;
						break;
					}
				}
			}
				
			// bord du bas
			if (!tile_calc[l_tile][c_tile] && l_tile < nb_tile - 1){ 
				local_l = TILE_SIZE - 1;
				l = l_tile * TILE_SIZE + local_l;
				for (local_c = 0; local_c < TILE_SIZE; local_c++){
					c = c_tile * TILE_SIZE + local_c;
					
					current_state = cur_img (l, c);
					cell_arround = number_arround_alive_cell (l, c);
					
					if ((current_state == DEAD_STATE  && cell_arround == 3) || // cell is dead and 3 cells alive arround
						(current_state == ALIVE_STATE && (cell_arround == 2 || cell_arround == 3))){ // cell is alive and 2 or 3 cells alive arround
						tile_calc[l_tile][c_tile] = true;
						break;
					}
				}
			}
		}
	}
}

// initialise la matrice qui associe les booleens à chaque tuile
void initialize_tile_calc (){
	int nb_tile = DIM / TILE_SIZE;
	if (tile_calc == NULL){
		tile_calc = malloc (nb_tile * sizeof (int*));
		for (int i = 0; i < nb_tile; i++){
			tile_calc[i] = malloc (nb_tile * sizeof (int));
			for (int j = 0; j < nb_tile; j++){
				tile_calc[i][j] = 1;		
			}
		}
	}
}

// free all data struct
void clean_compute (){
	int nb_tile = DIM / TILE_SIZE;
	if (tile_calc != NULL){
		for (int i = 0; i < nb_tile; i++){
			free(tile_calc[i]);
		}
		free(tile_calc);
	}
}

///////////////////////////// Version séquentielle simple

// version séquentielle de base
unsigned compute_seq_base (unsigned nb_iter) {
	int current_iter = 0;
	int next_image_change = false;

	for (unsigned it = 1; it <= nb_iter; it ++) {
		current_iter = it;
    	for (int l = 0; l < DIM; l++){
			for (int c = 0; c < DIM; c++){
				next_image_change = (set_next_state(l, c) || next_image_change);
			}
		}
		
		if (!next_image_change) {
			current_iter = it-1; 
			break;
		}
		else {
	    	swap_images ();
    	}
	}
	
	// retourne le nombre d'étapes nécessaires à la
    // stabilisation du calcul ou bien 0 si le calcul n'est pas
    // stabilisé au bout des nb_iter itérations
    return next_image_change ? 0 : current_iter;
}

// version séquentielle tuilée
unsigned compute_seq_tile (unsigned nb_iter) {
	int nb_tile = DIM / TILE_SIZE;

	int l, c;
	
	int current_iter = 0;
	int next_image_change = false;

	for (unsigned it = 1; it <= nb_iter; it ++) {
	
		// parcours de toute les tuiles
    	for (int l_tile = 0; l_tile < nb_tile; l_tile++){
			for (int c_tile = 0; c_tile < nb_tile; c_tile++){
				// parcours de toutes les cellules de chaques tuiles
				for (int local_l = 0; local_l < TILE_SIZE; local_l++){
					for (int local_c = 0; local_c < TILE_SIZE; local_c++){			
						l = l_tile * TILE_SIZE + local_l;
						c = c_tile * TILE_SIZE + local_c;
						
						next_image_change = (set_next_state(l, c) || next_image_change);
					}
				}
			}
		}
    
    	if (!next_image_change) {
			current_iter = it-1; 
			break;
		}
		else {
	    	swap_images ();
    	}
	}
	
	// retourne le nombre d'étapes nécessaires à la
    // stabilisation du calcul ou bien 0 si le calcul n'est pas
    // stabilisé au bout des nb_iter itérations
    return next_image_change ? 0 : current_iter;
}



unsigned compute_seq_tile_optimized (unsigned nb_iter){
	int nb_tile = DIM / TILE_SIZE;
	
	int l, c;
	
	int current_iter = 0;
	bool next_image_change = false;
	bool tile_change = false;
	
	initialize_tile_calc();

	for (unsigned it = 1; it <= nb_iter; it ++) {
	
		// parcours de toute les tuiles
    	for (int l_tile = 0; l_tile < nb_tile; l_tile++){
			for (int c_tile = 0; c_tile < nb_tile; c_tile++){
			
				if (tile_calc[l_tile][c_tile]){
					tile_change = false;
					// parcours de toutes les cellules de chaques tuiles
					for (int local_l = 0; local_l < TILE_SIZE; local_l++){
						for (int local_c = 0; local_c < TILE_SIZE; local_c++){			
							l = l_tile * TILE_SIZE + local_l;
							c = c_tile * TILE_SIZE + local_c;
						
					
							tile_change = (set_next_state(l, c) || tile_change);
						
							next_image_change = (tile_change || next_image_change);
						}
					}
					
					tile_calc[l_tile][c_tile] = tile_change;
				}
			}
		}
    
    	if (!next_image_change) {
			current_iter = it-1; 
			break;
		}
		else {
	    	swap_images ();
	    	update_tile_calc();
    	}
	}
	
	// retourne le nombre d'étapes nécessaires à la
    // stabilisation du calcul ou bien 0 si le calcul n'est pas
    // stabilisé au bout des nb_iter itérations
    return next_image_change ? 0 : current_iter;
}


///////////////////////////// Version OpenMP de base

// version OpenMP (for) de base
unsigned compute_omp_for_base (unsigned nb_iter) {
	int current_iter = 0;
	int next_image_change = true;

	for (unsigned it = 1; it <= nb_iter; it ++) {
		current_iter = it;
        #pragma omp parallel for
    	for (int l = 0; l < DIM; l++){
			for (int c = 0; c < DIM; c++){
				next_image_change = (set_next_state(l, c) || next_image_change);
			}
		}

		swap_images ();
	}
	
	// retourne le nombre d'étapes nécessaires à la
    // stabilisation du calcul ou bien 0 si le calcul n'est pas
    // stabilisé au bout des nb_iter itérations
    return next_image_change ? 0 : current_iter;
}

// version OpenMP (for) tuillée
unsigned compute_omp_for_tile (unsigned nb_iter) {
	int current_iter = 0;
	int next_image_change = true;

	for (unsigned it = 1; it <= nb_iter; it ++) {
		current_iter = it;
        #pragma omp parallel for collapse(2) schedule(dynamic, TILE_SIZE)
    	for (int l = 0; l < DIM; l++){
			for (int c = 0; c < DIM; c++){
				next_image_change = (set_next_state(l, c) || next_image_change);
			}
		}

		swap_images ();
	}
	
	// retourne le nombre d'étapes nécessaires à la
    // stabilisation du calcul ou bien 0 si le calcul n'est pas
    // stabilisé au bout des nb_iter itérations
    return next_image_change ? 0 : current_iter;
}



//////////////////////////////////////////////////////////////////////////////////


void first_touch_v1 ()
{
  int i,j ;

#pragma omp parallel for
  for(i=0; i<DIM ; i++) {
    for(j=0; j < DIM ; j += 512)
      next_img (i, j) = cur_img (i, j) = 0 ;
  }
}

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
/*unsigned compute_v1(unsigned nb_iter){
    //return compute_v1_base (nb_iter);
    return compute_v1_tile (nb_iter);
}*/



///////////////////////////// Version OpenMP optimisée

void first_touch_v2 ()
{

}

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v2(unsigned nb_iter)
{
  return 0; // on ne s'arrête jamais
}


///////////////////////////// Version OpenCL

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v3 (unsigned nb_iter)
{
  return ocl_compute (nb_iter);
}
