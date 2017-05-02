
#include "compute.h"
#include "graphics.h"
#include "debug.h"
#include "ocl.h"

#include <stdbool.h>

unsigned version = 0;

unsigned compute_seq_base (unsigned nb_iter);
unsigned compute_seq_tile (unsigned nb_iter);
unsigned compute_seq_tile_optimized (unsigned nb_iter);
unsigned compute_omp_for_base (unsigned nb_iter);
unsigned compute_omp_for_tile (unsigned nb_iter);
unsigned compute_omp_for_optimized (unsigned nb_iter);
unsigned compute_omp_task_tile (unsigned nb_iter);
unsigned compute_omp_task_optimized (unsigned nb_iter);
unsigned compute_v3 (unsigned nb_iter);

void_func_t first_touch [] = {
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
};

int_func_t compute [] = {
  compute_seq_base, // = -v 0
  compute_seq_tile,  // = -v 1 ...
  compute_seq_tile_optimized,
  compute_omp_for_base,
  compute_omp_for_tile,
  compute_omp_for_optimized,
  compute_omp_task_tile,
  compute_omp_task_optimized,
  compute_v3,
};

char *version_name [] = {
  "Séquentielle de base",
  "Séquentielle tuilée",
  "Séquentielle tuilée optimisée",
  "OpenMP (for) de base",
  "OpenMP (for) tuilée",
  "OpenMP (for) tuilée optimisée",
  "OpenMP (task) tuilée",
  "OpenMP (task) tuilée optimisée",
  "OpenCL",
};

unsigned opencl_used [] = {
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  1,
};

// Matrice qui va associer les coordonnées d'une tuile à un booléen
// Si booleen à vrai => la matrice est suceptible de changer sinon la tuile doit être ignorée
// 		* initialisée dans initialize_tile_calc() si sa valeur = NULL (= une fois dans l'exécution)
//		* clean dans clean_compute() (appelée à la fin du main)
//		* mis à jour à la fin de chaque itération par update_tile_calc()
int **tile_calc = NULL;

///////////////////////////// Fonctions

// Calcul le nombre de cellule vivante autour de la cellule (l, c)
int number_arround_alive_cell (int l, int c){
	int cpt_alive_cell = 0;
	
	// top
	if (l > 0) cpt_alive_cell += (cur_img (l-1, c) != DEAD_STATE);
	
	// top-right corner
	if (l > 0 && c < DIM-1) cpt_alive_cell += (cur_img (l-1, c+1) != DEAD_STATE);
	
	// right
	if (c < DIM-1) cpt_alive_cell += (cur_img (l, c+1) != DEAD_STATE);
	
	// right-bottom corner
	if (l < DIM-1 && c < DIM-1)	cpt_alive_cell += (cur_img (l+1, c+1) != DEAD_STATE);
	
	// bottom
	if (l < DIM-1) cpt_alive_cell += (cur_img (l+1, c) != DEAD_STATE);
	
	// left-bottom corner
	if (l < DIM-1 && c > 0)	cpt_alive_cell += (cur_img (l+1, c-1) != DEAD_STATE);
	
	// left
	if (c > 0) cpt_alive_cell += (cur_img (l, c-1) != DEAD_STATE);
	
	// top-left corner
	if (l > 0 && c > 0)	cpt_alive_cell += (cur_img (l-1, c-1) != DEAD_STATE);
	
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

// retourne vrai si la tuile de coordonnées (l_tile, c_tile) va changer au prochain calcul
// 		* static_field = 0 -> ligne constante avec une valeur de value_static_field
// 		* static_field != 0 -> colonne constante avec une valeur de value_static_field
bool check_boarder (int l_tile, int c_tile, int static_field, int value_static_field){
	int l = 0, c = 0;
	
	int cell_arround;
	Uint32 current_state;
	
	bool ret = false;
	
	if (static_field == 0){
		l = l_tile * TILE_SIZE + value_static_field;
	}
	else {
		c = c_tile * TILE_SIZE + value_static_field;
	}
	
	for (int variable_field = 0; variable_field < TILE_SIZE; variable_field++){
		if (static_field != 0){
			l = l_tile * TILE_SIZE + variable_field;
		}
		else {
			c = c_tile * TILE_SIZE + variable_field;
		}
		
		current_state = cur_img (l, c);
		cell_arround = number_arround_alive_cell (l, c);
		
		if ((current_state == DEAD_STATE  && cell_arround == 3) || // cell is dead and 3 cells alive arround
			(current_state == ALIVE_STATE && (cell_arround == 2 || cell_arround == 3))){ // cell is alive and 2 or 3 cells alive arround
			ret = true;
			break;
		}
	}
	
	return ret;
}

// Met à jour la matrice qui associe les tuiles à un booleen
void update_tile_calc_seq (){
	int nb_tile = DIM / TILE_SIZE;
	
	// parcours de toute les tuiles
	for (int l_tile = 0; l_tile < nb_tile; l_tile++){
		for (int c_tile = 0; c_tile < nb_tile; c_tile++){
			
			// Si la tuile va être calculé au prochain tour on ignore ses vérifications
			if (!tile_calc[l_tile][c_tile]){
				tile_calc[l_tile][c_tile] =
					 // Si la tuile est ignorée
					 // 	&& si elle n'est pas sur la première ligne
					 // 	&& si un pixel sur ça bordure haute (première ligne de la tuile) va changer au prochain calcul
					 // => on met le booleen a true dans la matrice et on arrête les vérifications
					 (l_tile > 0 && check_boarder (l_tile, c_tile, 0, 0)) ||
					 
					 // OU Si la tuile est ignorée
					 // 	&& si elle n'est pas sur la première colonne
					 // 	&& si un pixel sur ça bordure à gauche (première colonne de la tuile) va changer au prochain calcul
					 // => on met le booleen a true dans la matrice et on arrête les vérifications
					 (!tile_calc[l_tile][c_tile] && c_tile > 0 && check_boarder (l_tile, c_tile, 1, 0)) ||
					 
					 // OU Si la tuile est ignorée
					 // 	&& si elle n'est pas sur la dernière ligne
					 // 	&& si un pixel sur ça bordure basse (dernière ligne de la tuile) va changer au prochain calcul
					 // => on met le booleen a true dans la matrice et on arrête les vérifications
					 (!tile_calc[l_tile][c_tile] && (l_tile < nb_tile - 1) && check_boarder (l_tile, c_tile, 0, TILE_SIZE - 1)) ||
					 
					 // OU Si la tuile est ignorée
					 // 	&& si elle n'est pas sur la dernière colonne
					 // 	&& si un pixel sur ça bordure à droite (dernière colonne de la tuile) va changer au prochain calcul
					 // => on met le booleen a true dans la matrice
					 (!tile_calc[l_tile][c_tile] && (c_tile < nb_tile - 1) && check_boarder (l_tile, c_tile, 1, TILE_SIZE - 1));
			 }
		}
	}
}

void update_tile_calc_omp_for (){
	int nb_tile = DIM / TILE_SIZE;
	
	// parcours de toute les tuiles
	#pragma omp parallel for
	for (int l_tile = 0; l_tile < nb_tile; l_tile++){
		for (int c_tile = 0; c_tile < nb_tile; c_tile++){
		
			// Si la tuile va être calculé au prochain tour on ignore ses vérifications
			if (!tile_calc[l_tile][c_tile]){
				tile_calc[l_tile][c_tile] =
					 // Si la tuile est ignorée
					 // 	&& si elle n'est pas sur la première ligne
					 // 	&& si un pixel sur ça bordure haute (première ligne de la tuile) va changer au prochain calcul
					 // => on met le booleen a true dans la matrice et on arrête les vérifications
					 (l_tile > 0 && check_boarder (l_tile, c_tile, 0, 0)) ||
					 
					 // OU Si la tuile est ignorée
					 // 	&& si elle n'est pas sur la première colonne
					 // 	&& si un pixel sur ça bordure à gauche (première colonne de la tuile) va changer au prochain calcul
					 // => on met le booleen a true dans la matrice et on arrête les vérifications
					 (!tile_calc[l_tile][c_tile] && c_tile > 0 && check_boarder (l_tile, c_tile, 1, 0)) ||
					 
					 // OU Si la tuile est ignorée
					 // 	&& si elle n'est pas sur la dernière ligne
					 // 	&& si un pixel sur ça bordure basse (dernière ligne de la tuile) va changer au prochain calcul
					 // => on met le booleen a true dans la matrice et on arrête les vérifications
					 (!tile_calc[l_tile][c_tile] && (l_tile < nb_tile - 1) && check_boarder (l_tile, c_tile, 0, TILE_SIZE - 1)) ||
					 
					 // OU Si la tuile est ignorée
					 // 	&& si elle n'est pas sur la dernière colonne
					 // 	&& si un pixel sur ça bordure à droite (dernière colonne de la tuile) va changer au prochain calcul
					 // => on met le booleen a true dans la matrice
					 (!tile_calc[l_tile][c_tile] && (c_tile < nb_tile - 1) && check_boarder (l_tile, c_tile, 1, TILE_SIZE - 1));
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
			current_iter = it; 
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
			current_iter = it; 
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

// Version séquentielle tuilée optmisée
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
			current_iter = it; 
			break;
		}
		else {
	    	swap_images ();
	    	update_tile_calc_seq();
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

		if (!next_image_change) {
			current_iter = it; 
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

// version OpenMP (for) tuilée
unsigned compute_omp_for_tile (unsigned nb_iter) {
	int current_iter = 0;
	int next_image_change = true;

	for (unsigned it = 1; it <= nb_iter; it ++) {
		current_iter = it;
        #pragma omp parallel for collapse(2) schedule(static, TILE_SIZE)
    	for (int l = 0; l < DIM; l++){
			for (int c = 0; c < DIM; c++){
				next_image_change = (set_next_state(l, c) || next_image_change);
			}
		}

		if (!next_image_change) {
			current_iter = it; 
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

// version OpenMP (for) tuilée optimisée
unsigned compute_omp_for_optimized(unsigned nb_iter)
{
	int current_iter = 0;
	int next_image_change = true;

	for (unsigned it = 1; it <= nb_iter; it ++) {
		current_iter = it;
        #pragma omp parallel for collapse(2) schedule(dynamic, TILE_SIZE)
    	for (int l = 0; l < DIM; l++){
			for (int c = 0; c < DIM; c++){
				if (tile_calc[l][c])
					next_image_change = (set_next_state(l, c) || next_image_change);
			}
		}

		if (!next_image_change) {
			current_iter = it; 
			break;
		}
		else {
	    	swap_images ();
			update_tile_calc_omp_for();
    	}
	}
	
	// retourne le nombre d'étapes nécessaires à la
    // stabilisation du calcul ou bien 0 si le calcul n'est pas
    // stabilisé au bout des nb_iter itérations
    return next_image_change ? 0 : current_iter;
}

// Version OpenMP (task) tuilée
unsigned compute_omp_task_tile(unsigned nb_iter)
{
	int nb_tile = DIM / TILE_SIZE;
	int current_iter = 0;
	int next_image_change = true;
	
	int l, c;

	for (unsigned it = 1; it <= nb_iter; it ++) {
		current_iter = it;
		
		#pragma omp parallel
		{
			#pragma omp master
			for (int l_tile = 0; l_tile < nb_tile; l_tile++){
				for (int c_tile = 0; c_tile < nb_tile; c_tile++){
						
					#pragma omp task private(l,c)
					for (int local_l = 0; local_l < TILE_SIZE; local_l++){
						for (int local_c = 0; local_c < TILE_SIZE; local_c++){			
							l = l_tile * TILE_SIZE + local_l;
							c = c_tile * TILE_SIZE + local_c;
						
							next_image_change = (set_next_state(l, c) || next_image_change);
						}
					}
				}
			}
		}

		// barrier
		#pragma omp taskwait

		if (!next_image_change) {
			current_iter = it; 
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

// Version OpenMP (task) tuilée optimisée
unsigned compute_omp_task_optimized(unsigned nb_iter)
{
	int nb_tile = DIM / TILE_SIZE;
	int current_iter = 0;
	int next_image_change = true;
	int tile_change = false;
	
	int l,c;
	
	initialize_tile_calc();

	for (unsigned it = 1; it <= nb_iter; it ++) {
		current_iter = it;
		
		#pragma omp parallel
		{
		    #pragma omp master
			for (int l_tile = 0; l_tile < nb_tile; l_tile++){
				for (int c_tile = 0; c_tile < nb_tile; c_tile++){
						
					if (tile_calc[l_tile][c_tile]) {
						#pragma omp task private(l,c) firstprivate(tile_change)
						{
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
			}
		}
		
		// barrier
		#pragma omp taskwait

		if (!next_image_change) {
			current_iter = it; 
			break;
		}
		else {
	    	swap_images ();
	    	update_tile_calc_omp_for();
    	}
	}
	// retourne le nombre d'étapes nécessaires à la
    // stabilisation du calcul ou bien 0 si le calcul n'est pas
    // stabilisé au bout des nb_iter itérations
    return next_image_change ? 0 : current_iter;
}

///////////////////////////// Version OpenCL

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v3 (unsigned nb_iter)
{
 return ocl_compute (nb_iter);
}
