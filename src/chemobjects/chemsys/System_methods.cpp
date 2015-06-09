#include "System.h"


namespace libchemobjects{
namespace libchemsys{


//================= ObjectSpace member-functions =========================

int System::get_atom_index_by_atom_id(int id){
/*****************************************************************
 This auxiliary function returns the index of the atom given the id
 of the atom. If no atom with given id were found -1 is returned.
*****************************************************************/
  int indx = -1;
  for(int i=0;i<Number_of_atoms;i++){
    if(Atoms[i].is_Atom_id){ if(Atoms[i].Atom_id==id){indx = i; }  }
  }
  return indx;
}

int System::get_fragment_index_by_fragment_id(int id){
/*****************************************************************
 This auxiliary function returns the index of the fragment given the id
 of the fragment. If no fragment with given id were found -1 is returned.
*****************************************************************/
  int indx = -1;
  for(int i=0;i<Number_of_fragments;i++){
    if(Fragments[i].is_Group_id){ if(Fragments[i].Group_id==id){indx = i; }  }
  }
  return indx;
}

int System::get_molecule_index_by_molecule_id(int id){
/*****************************************************************
 This auxiliary function returns the index of the molecule given the id
 of the molecule. If no molecule with given id were found -1 is returned.
*****************************************************************/
  int indx = -1;
  for(int i=0;i<Number_of_molecules;i++){
    if(Molecules[i].is_Molecule_id){ if(Molecules[i].Molecule_id==id){indx = i; }  }
  }
  return indx;
}


int System::Find_Bond(int at_indx1,int at_indx2){
/*****************************************************************
   This class System method is looking for bond formed by 2 atoms 
   with globAtom_Indexes given by at_indx1, at_indx2
   It returns globGroup_Index of corresponding bond
******************************************************************/
  int res = -1;
  for(int i=0;i<Number_of_bonds;i++){
    if((  (Bonds[i].globAtom_Index[0]==at_indx1) && (Bonds[i].globAtom_Index[1]==at_indx2)  )||
       (  (Bonds[i].globAtom_Index[0]==at_indx2) && (Bonds[i].globAtom_Index[1]==at_indx1)  )
      ){   res = i;break;   }
  }
  return res;
}

int System::Find_Frag_Pair(int at_indx1,int at_indx2){
/*****************************************************************
   This class System method is looking for bond formed by 2 atoms
   with globAtom_Indexes given by at_indx1, at_indx2
   It returns globGroup_Index of corresponding bond
******************************************************************/
  int res = -1;
  for(int i=0;i<Number_of_frag_pairs;i++){
    int fp = Frag_pairs[i];
    if((  (Pairs[fp].globAtom_Index[0]==at_indx1) && (Pairs[fp].globAtom_Index[1]==at_indx2)  )||
       (  (Pairs[fp].globAtom_Index[0]==at_indx2) && (Pairs[fp].globAtom_Index[1]==at_indx1)  )
      ){   res = i;break;   }
  }
  return res;
}

int System::Find_Angle(int at_indx1,int at_indx3){
/*****************************************************************
   This class System method is looking for angle formed by 3 atoms
   with globAtom_Indexes given by at_indx1, (any) at_indx3
   It returns globGroup_Index of corresponding angle
******************************************************************/
  int res = -1;
  for(int i=0;i<Number_of_angles;i++){
    if((  (Angles[i].globAtom_Index[0]==at_indx1) && (Angles[i].globAtom_Index[2]==at_indx3)  )||
       (  (Angles[i].globAtom_Index[0]==at_indx3) && (Angles[i].globAtom_Index[2]==at_indx1)  )
      ){   res = i;break;   }
  }
  return res;
}

int System::Find_Angle(int at_indx1,int at_indx2,int at_indx3){
/*****************************************************************
   This class System method is looking for angle formed by 3 atoms
   with globAtom_Indexes given by at_indx1, at_indx2, at_indx3
   It returns globGroup_Index of corresponding angle
******************************************************************/
  int res = -1;
  for(int i=0;i<Number_of_angles;i++){
    if((  (Angles[i].globAtom_Index[0]==at_indx1) && (Angles[i].globAtom_Index[1]==at_indx2) && (Angles[i].globAtom_Index[2]==at_indx3) )||
       (  (Angles[i].globAtom_Index[0]==at_indx3) && (Angles[i].globAtom_Index[1]==at_indx2) && (Angles[i].globAtom_Index[2]==at_indx1) )
      ){   res = i;break;   }
  }
  return res;
}

int System::Find_Dihedral(int at_indx1,int at_indx2,int at_indx3,int at_indx4){
/*****************************************************************
   This class System method looking for dihedral formed by 4 atoms
   with globAtom_Indexes given by at_indx1, at_indx2, at_indx3, at_indx4
   It returns globGroup_Index of corresponding dihedral
******************************************************************/
  int res = -1;
  for(int i=0;i<Number_of_dihedrals;i++){
    if((  (Dihedrals[i].globAtom_Index[0]==at_indx1)
        &&(Dihedrals[i].globAtom_Index[1]==at_indx2)
        &&(Dihedrals[i].globAtom_Index[2]==at_indx3)
        &&(Dihedrals[i].globAtom_Index[3]==at_indx4)
       )||
       (  (Dihedrals[i].globAtom_Index[0]==at_indx4)
        &&(Dihedrals[i].globAtom_Index[1]==at_indx3)
        &&(Dihedrals[i].globAtom_Index[2]==at_indx2)
        &&(Dihedrals[i].globAtom_Index[3]==at_indx1)
       )
      ){   res = i;break;   }
  }
  return res;
}

int System::Find_Improper(int at_indx1){
/*****************************************************************
   This class System method is looking for an improper with a
   central atom having index at_indx1
   It returns globGroup_Index of corresponding improper
******************************************************************/
  int res = -1;
  for(int i=0;i<Number_of_impropers;i++){
    if(Impropers[i].globAtom_Index[0]==at_indx1 ){   res = i;break;   }
  }
  return res;
}

int System::is_12pair(int at_indx1,int at_indx2){
/*******************************************************************
  This function checks if the atoms with indexes at_indx1 and at_indx2
  are directly connected to each other
  Return 1 if they are connected to each other
         0 otherwise
*******************************************************************/
  int res = 0;
  for(int i=0;i<Atoms[at_indx1].globAtom_Adjacent_Atoms.size();i++){
    if(Atoms[at_indx1].globAtom_Adjacent_Atoms[i]==at_indx2){ res = 1; }
  }
  return res;
}

int System::is_13pair(int at_indx1,int at_indx2){
/*******************************************************************
  This function checks if the atoms with indexes at_indx1 and at_indx2
  are connected to common atom (form 1,3-pair)
  Return 1 if they are connected common atom
         0 otherwise
*******************************************************************/
  int res = 0;
  for(int i=0;i<Atoms[at_indx1].globAtom_Adjacent_Atoms.size();i++){
    for(int j=0;j<Atoms[at_indx2].globAtom_Adjacent_Atoms.size();j++){
      if(Atoms[at_indx1].globAtom_Adjacent_Atoms[i]==Atoms[at_indx2].globAtom_Adjacent_Atoms[j]){ res = 1; }
    }
  }
  return res;
}

int System::is_14pair(int at_indx1,int at_indx2){
/*******************************************************************
  This function checks if the atoms with indexes at_indx1 and at_indx2
  are connected to some other atoms which are directly conneted to 
  each other (form 1,4-pair)
  Return 1 if they form 1,4-pair
         0 otherwise
*******************************************************************/
  int res = 0;
  for(int i=0;i<Atoms[at_indx1].globAtom_Adjacent_Atoms.size();i++){
    for(int j=0;j<Atoms[at_indx2].globAtom_Adjacent_Atoms.size();j++){
      if(is_12pair(Atoms[at_indx1].globAtom_Adjacent_Atoms[i],Atoms[at_indx2].globAtom_Adjacent_Atoms[j])){ res = 1; }
    }
  }
  return res;
}

int System::is_group_pair(int at_indx1,int at_indx2){
/***********************************************************************
  This function checks if the atoms with indexes at_indx1 and at_indx
  belong to the same group(fragment)
  Return 1 if they belong to the same group
         0 otherwise
***********************************************************************/
  return (Atoms[at_indx1].globGroup_Index == Atoms[at_indx2].globGroup_Index);
}


void System::show_atoms(){
  for(int i=0;i<Number_of_atoms;i++){ Atoms[i].show_info();  }
}

void System::show_bonds(){
  for(int i=0;i<Number_of_bonds;i++){ Bonds[i].show_info();  }
}

void System::show_angles(){
  for(int i=0;i<Number_of_angles;i++){ Angles[i].show_info(); }
}

void System::show_dihedrals(){
  for(int i=0;i<Number_of_dihedrals;i++){ Dihedrals[i].show_info(); }
}

void System::show_impropers(){
  for(int i=0;i<Number_of_impropers;i++){ Impropers[i].show_info();  }
}

void System::show_pairs(){
  for(int i=0;i<Number_of_pairs;i++){ Pairs[i].show_info();  }
}

void System::show_frag_bonds(){
  for(int i=0;i<Number_of_frag_bonds;i++){ Bonds[Frag_bonds[i]].show_info();  }
}

void System::show_frag_angles(){
  for(int i=0;i<Number_of_frag_angles;i++){ Angles[Frag_angles[i]].show_info(); }
}

void System::show_frag_dihedrals(){
  for(int i=0;i<Number_of_frag_dihedrals;i++){ Dihedrals[Frag_dihedrals[i]].show_info(); }
}

void System::show_frag_impropers(){
  for(int i=0;i<Number_of_frag_impropers;i++){ Impropers[Frag_impropers[i]].show_info();  }
}

void System::show_frag_pairs(){
  for(int i=0;i<Number_of_frag_pairs;i++){ Pairs[Frag_pairs[i]].show_info();  }
} 

void System::show_fragments(){
  for(int i=0;i<Number_of_fragments;i++){ Fragments[i].show_info();  }
}

void System::show_rings(){
  for(int i=0;i<Number_of_rings;i++){ Rings[i].show_info();  }
}

void System::show_molecules(){
  for(int i=0;i<Number_of_molecules;i++){ Molecules[i].show_info();  }
}

void System::show_interactions(){
//  for(int i=0;i<interactions.size();i++){ interactions[i].show_info();  }
}


void System::show_interactions(std::string s_int_type){
  int int_type = -1;
  if(s_int_type=="bond")         { int_type = 0; }
  else if(s_int_type=="angle")   { int_type = 1; }
  else if(s_int_type=="dihedral"){ int_type = 2; }
  else if(s_int_type=="oop")     { int_type = 3; }
  else if(s_int_type=="vdw")     { int_type = 4; }
  else if(s_int_type=="elec")    { int_type = 5; }
  else if(s_int_type=="mb")      { int_type = 6; }
  else if(s_int_type=="cg")      { int_type = 7; }

//  for(int i=0;i<interactions.size();i++){
//    if(interactions[i].get_type()==int_type){    interactions[i].show_info();    }
//  }

}



}// namespace libchemsys
}// namespace libchemobjects
