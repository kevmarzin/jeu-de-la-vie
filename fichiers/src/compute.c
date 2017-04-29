
#include "compute.h"
#include "graphics.h"
#include "debug.h"
#include "ocl.h"

#include <stdbool.h>

unsigned version = 0;

void first_touch_v1 (void);
void first_touch_v2 (void);

unsigned compute_v0 (unsigned nb_iter);
unsigned compute_v1 (unsigned nb_iter);
unsigned compute_v2 (unsigned nb_iter);
unsigned compute_v3 (unsigned nb_iter);

void_func_t first_touch [] = {
  NULL,
  first_touch_v1,
  first_touch_v2,
  NULL,
};

int_func_t compute [] = {
  compute_v0,
  compute_v1,
  compute_v2,
  compute_v3,
};

char *version_name [] = {
  "Séquentielle",
  "OpenMP",
  "OpenMP zone",
  "OpenCL",
};

unsigned opencl_used [] = {
  0,
  0,
  0,
  1,
};

///////////////////////////// Fonctions

// Calcul le nombre de cellule vivante autour d'une cellule gràce à ses coordonnées
int number_arround_alive_cell (int i, int j, Uint32 dead_state){
	int cpt_alive_cell = 0;
	
	// top
	if (i > 0) cpt_alive_cell += (cur_img (i-1, j) != dead_state);
	
	// top-right corner
	if (i > 0 && j < DIM-1) cpt_alive_cell += (cur_img (i-1, j+1) != dead_state);
	
	// right
	if (j < DIM-1) cpt_alive_cell += (cur_img (i, j+1) != dead_state);
	
	// right-bottom corner
	if (i < DIM-1 && j < DIM-1)	cpt_alive_cell += (cur_img (i+1, j+1) != dead_state);
	
	// bottom
	if (i < DIM-1) cpt_alive_cell += (cur_img (i+1, j) != dead_state);
	
	// left-bottom corner
	if (i < DIM-1 && j > 0)	cpt_alive_cell += (cur_img (i+1, j-1) != dead_state);
	
	// left
	if (j > 0) cpt_alive_cell += (cur_img (i, j-1) != dead_state);
	
	// top-left corner
	if (i > 0 && j > 0)	cpt_alive_cell += (cur_img (i-1, j-1) != dead_state);
	
	return cpt_alive_cell;
	
}

// Calcul le prochain état du pixel ligne l et colonne c
Uint32 get_next_state (int l, int c) {
    
    Uint32 dead_state  = 0x0;
    Uint32 alive_state = 0xffff00ff;
    
    int cell_arround = number_arround_alive_cell (l, c, dead_state);
    
    Uint32 current_state = cur_img (l, c);
			
    bool alive_on_next_step =
        ((current_state == dead_state  && cell_arround == 3)  || // cell is dead and 3 cells alive arround
            (current_state == alive_state && (cell_arround == 2 || cell_arround == 3))); // cell is alive and 2 or 3 cells alive arround

    return (alive_on_next_step ? alive_state : dead_state);
}

///////////////////////////// Version séquentielle simple

// version séquentielle de base
unsigned compute_v0_base (unsigned nb_iter) {
	Uint32 next_state;
	
	int current_iter = 0;
	int next_image_change = false;

	for (unsigned it = 1; it <= nb_iter; it ++) {
		current_iter = it;
    	for (int l = 0; l < DIM; l++){
			for (int c = 0; c < DIM; c++){
				next_img (l, c) = (next_state = get_next_state(l, c));
				
				if (!next_image_change && cur_img (l, c) != next_state){
					next_image_change = true;
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

// version séquentielle tuilée
unsigned compute_v0_tile (unsigned nb_iter) {
	Uint32 next_state;
	
	int size_tile = 32;
	
	int l, c;
	
	int current_iter = 0;
	int next_image_change = false;

	for (unsigned it = 1; it <= nb_iter; it ++) {
	
		// parcours de toute les tuiles
    	for (int l_tile = 0; l_tile < DIM / size_tile; l_tile++){
			for (int c_tile = 0; c_tile < DIM / size_tile; c_tile++){
				// parcours de toutes les cellules de chaques tuiles
				for (int local_l = 0; local_l < size_tile; local_l++){
					for (int local_c = 0; local_c < size_tile; local_c++){			
						l = l_tile * size_tile + local_l;
						c = c_tile * size_tile + local_c;
						
						next_img (l, c) = (next_state = get_next_state(l, c));
				
                        if (!next_image_change && cur_img (l, c) != next_state){
                            next_image_change = true;
                        }
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



unsigned compute_v0 (unsigned nb_iter){
//	return compute_v0_base (nb_iter);
	return compute_v0_tile (nb_iter);
}


///////////////////////////// Version OpenMP de base

// version OpenMP de base
unsigned compute_v1_base (unsigned nb_iter) {
	Uint32 next_state;
	
	int current_iter = 0;
	int next_image_change = true;

	for (unsigned it = 1; it <= nb_iter; it ++) {
		current_iter = it;
        #pragma omp parallel for
    	for (int l = 0; l < DIM; l++){
			for (int c = 0; c < DIM; c++){
				next_img (l, c) = (next_state = get_next_state(l, c));
				
				if (!next_image_change && cur_img (l, c) != next_state){
					next_image_change = true;
				}
			}
			
		}

		swap_images ();

	}
	
	// retourne le nombre d'étapes nécessaires à la
    // stabilisation du calcul ou bien 0 si le calcul n'est pas
    // stabilisé au bout des nb_iter itérations
    return next_image_change ? 0 : current_iter;
}

// version séquentielle tuillée
unsigned compute_v1_tile (unsigned nb_iter) {
	Uint32 next_state;
	
	int current_iter = 0;
	int next_image_change = true;

	for (unsigned it = 1; it <= nb_iter; it ++) {
		current_iter = it;
        #pragma omp parallel for collapse(2) schedule(dynamic, 32)
    	for (int l = 0; l < DIM; l++){
			for (int c = 0; c < DIM; c++){
				next_img (l, c) = (next_state = get_next_state(l, c));
				
				if (!next_image_change && cur_img (l, c) != next_state){
					next_image_change = true;
				}
			}
		}

		swap_images ();

	}
	
	// retourne le nombre d'étapes nécessaires à la
    // stabilisation du calcul ou bien 0 si le calcul n'est pas
    // stabilisé au bout des nb_iter itérations
    return next_image_change ? 0 : current_iter;
}

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
unsigned compute_v1(unsigned nb_iter){
    //return compute_v1_base (nb_iter);
    return compute_v1_tile (nb_iter);
}



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
