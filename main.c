static char help[] = "petsc-example-SEM ordering";

#include<petscdmplex.h>
#include <petscvec.h>


typedef struct{
  char          filename[PETSC_MAX_PATH_LEN]; //optional exodusII file
  //PetscInt      dof;                          //dof per node
  PetscInt      dim; /* Topological problem dimension */
  PetscInt      Nf;  /* Number of fields */
  PetscInt     *Nc;  /* Number of components per field */
  PetscInt     *k;   /* Spectral order per field */
}AppCtx;


#undef __FUNCT__
#define __FUNCT__ "processUserOptions"
PetscErrorCode processUserOptions(MPI_Comm comm, AppCtx *userOptions)
{

  PetscErrorCode	ierr;
  PetscBool       fileflg;
  PetscBool       flg;
  PetscInt        len;

  PetscFunctionBeginUser;
  userOptions->dim = 3;
  userOptions->Nf  = 0;
  userOptions->Nc  = NULL;
  userOptions->k   = NULL;
    ierr = PetscOptionsBegin(comm, "", "options", "DMPLEX");CHKERRQ(ierr);
      ierr = PetscOptionsString("-f", "Exodus.II filename to read", "main.c", userOptions->filename, userOptions->filename, sizeof(userOptions->filename), &fileflg);CHKERRQ(ierr);
      #if !defined(PETSC_HAVE_EXODUSII)
        if(flg)  SETERRQ(comm, PETSC_ERR_ARG_WRONG, "This option requires ExodusII support. Reconfigure your Arch with --download-exodusii");
      #endif
      //ERROR if no mesh file is provided
      if(fileflg == PETSC_FALSE)
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "This code requires a mesh file (exodusII)");

      ierr = PetscOptionsInt("-dim", "dimension of the mesh", "main", userOptions->dim, &userOptions->dim, NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-num_fields", "The number of fields", "petsc-example-SEM", userOptions->Nf, &userOptions->Nf, NULL);CHKERRQ(ierr);
      if (userOptions->Nf) {
        len  = userOptions->Nf;
        ierr = PetscMalloc1(len, &userOptions->Nc);CHKERRQ(ierr);
        ierr = PetscOptionsIntArray("-num_components", "The number of components per field", "ex6.c", userOptions->Nc, &len, &flg);CHKERRQ(ierr);
      if (flg && (len != userOptions->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Length of components array is %d should be %d", len, userOptions->Nf);
        len  = userOptions->Nf;
        ierr = PetscMalloc1(len, &userOptions->k);CHKERRQ(ierr);
        ierr = PetscOptionsIntArray("-order", "The spectral order per field", "petsc-example-SEM", userOptions->k, &len, &flg);CHKERRQ(ierr);
      if (flg && (len != userOptions->Nf)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Length of order array is %d should be %d", len, userOptions->Nf);
    }
    PetscOptionsEnd();
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "createDistributedDM"
PetscErrorCode createDistributedDM(MPI_Comm comm, AppCtx user, DM *dm){

  PetscErrorCode  ierr;
  const char      *filename = user.filename;
  PetscBool       interpolate = PETSC_FALSE;
  DM              distributedMesh = NULL;


  PetscFunctionBeginUser;
  if(user.k[0] >= 2){
      interpolate = PETSC_TRUE;
}
    ierr = DMPlexCreateFromFile(comm, filename, interpolate, dm);CHKERRQ(ierr);
    ierr = DMPlexDistribute(*dm, 0, NULL, &distributedMesh);CHKERRQ(ierr);
  	if (distributedMesh) {
  		ierr = DMDestroy(dm);CHKERRQ(ierr);
  		*dm  = distributedMesh;
  	}
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SetSymmetries"
PetscErrorCode SetSymmetries(DM dm, PetscSection s, AppCtx *user)
{
  PetscInt       f, o, i, j, k, c, d;
  DMLabel        depthLabel;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetLabel(dm,"depth",&depthLabel);CHKERRQ(ierr);
  for (f = 0; f < user->Nf; f++) {
    PetscSectionSym sym;

    if (user->k[f] < 3) continue; /* No symmetries needed for order < 3, because no cell, facet, edge or vertex has more than one node */
    ierr = PetscSectionSymCreateLabel(PetscObjectComm((PetscObject)s),depthLabel,&sym);CHKERRQ(ierr);

    for (d = 0; d <= user->dim; d++) {
      if (d == 1) {
        PetscInt        numDof  = user->k[f] - 1;
        PetscInt        numComp = user->Nc[f];
        PetscInt        minOrnt = -2;
        PetscInt        maxOrnt = 2;
        PetscInt        **perms;

        ierr = PetscCalloc1(maxOrnt - minOrnt,&perms);CHKERRQ(ierr);
        for (o = minOrnt; o < maxOrnt; o++) {
          PetscInt *perm;

          if (o == -1 || !o) { /* identity */
            perms[o - minOrnt] = NULL;
          } else {
            ierr = PetscMalloc1(numDof * numComp, &perm);CHKERRQ(ierr);
            for (i = numDof - 1, k = 0; i >= 0; i--) {
              for (j = 0; j < numComp; j++, k++) perm[k] = i * numComp + j;
            }
            perms[o - minOrnt] = perm;
          }
        }
        ierr = PetscSectionSymLabelSetStratum(sym,d,numDof*numComp,minOrnt,maxOrnt,PETSC_OWN_POINTER,(const PetscInt **) perms,NULL);CHKERRQ(ierr);
      } else if (d == 2) {
        PetscInt        perEdge = user->k[f] - 1;
        PetscInt        numDof  = perEdge * perEdge;
        PetscInt        numComp = user->Nc[f];
        PetscInt        minOrnt = -4;
        PetscInt        maxOrnt = 4;
        PetscInt        **perms;

        ierr = PetscCalloc1(maxOrnt-minOrnt,&perms);CHKERRQ(ierr);
        for (o = minOrnt; o < maxOrnt; o++) {
          PetscInt *perm;

          if (!o) continue; /* identity */
          ierr = PetscMalloc1(numDof * numComp, &perm);CHKERRQ(ierr);
          /* We want to perm[k] to list which *localArray* position the *sectionArray* position k should go to for the given orientation*/
          switch (o) {
          case 0:
            break; /* identity */
          case -4: /* flip along (-1,-1)--( 1, 1), which swaps edges 0 and 3 and edges 1 and 2.  This swaps the i and j variables */
            for (i = 0, k = 0; i < perEdge; i++) {
              for (j = 0; j < perEdge; j++, k++) {
                for (c = 0; c < numComp; c++) {
                  perm[k * numComp + c] = (perEdge * j + i) * numComp + c;
                }
              }
            }
            break;
          case -3: /* flip along (-1, 0)--( 1, 0), which swaps edges 0 and 2.  This reverses the i variable */
            for (i = 0, k = 0; i < perEdge; i++) {
              for (j = 0; j < perEdge; j++, k++) {
                for (c = 0; c < numComp; c++) {
                  perm[k * numComp + c] = (perEdge * (perEdge - 1 - i) + j) * numComp + c;
                }
              }
            }
            break;
          case -2: /* flip along ( 1,-1)--(-1, 1), which swaps edges 0 and 1 and edges 2 and 3.  This swaps the i and j variables and reverse both */
            for (i = 0, k = 0; i < perEdge; i++) {
              for (j = 0; j < perEdge; j++, k++) {
                for (c = 0; c < numComp; c++) {
                  perm[k * numComp + c] = (perEdge * (perEdge - 1 - j) + (perEdge - 1 - i)) * numComp + c;
                }
              }
            }
            break;
          case -1: /* flip along ( 0,-1)--( 0, 1), which swaps edges 3 and 1.  This reverses the j variable */
            for (i = 0, k = 0; i < perEdge; i++) {
              for (j = 0; j < perEdge; j++, k++) {
                for (c = 0; c < numComp; c++) {
                  perm[k * numComp + c] = (perEdge * i + (perEdge - 1 - j)) * numComp + c;
                }
              }
            }
            break;
          case  1: /* rotate section edge 1 to local edge 0.  This swaps the i and j variables and then reverses the j variable */
            for (i = 0, k = 0; i < perEdge; i++) {
              for (j = 0; j < perEdge; j++, k++) {
                for (c = 0; c < numComp; c++) {
                  perm[k * numComp + c] = (perEdge * (perEdge - 1 - j) + i) * numComp + c;
                }
              }
            }
            break;
          case  2: /* rotate section edge 2 to local edge 0.  This reverse both i and j variables */
            for (i = 0, k = 0; i < perEdge; i++) {
              for (j = 0; j < perEdge; j++, k++) {
                for (c = 0; c < numComp; c++) {
                  perm[k * numComp + c] = (perEdge * (perEdge - 1 - i) + (perEdge - 1 - j)) * numComp + c;
                }
              }
            }
            break;
          case  3: /* rotate section edge 3 to local edge 0.  This swaps the i and j variables and then reverses the i variable */
            for (i = 0, k = 0; i < perEdge; i++) {
              for (j = 0; j < perEdge; j++, k++) {
                for (c = 0; c < numComp; c++) {
                  perm[k * numComp + c] = (perEdge * j + (perEdge - 1 - i)) * numComp + c;
                }
              }
            }
            break;
          default:
            break;
          }
          perms[o - minOrnt] = perm;
        }
        ierr = PetscSectionSymLabelSetStratum(sym,d,numDof*numComp,minOrnt,maxOrnt,PETSC_OWN_POINTER,(const PetscInt **) perms,NULL);CHKERRQ(ierr);
      }
    }
    ierr = PetscSectionSetFieldSym(s,f,sym);CHKERRQ(ierr);
    ierr = PetscSectionSymDestroy(&sym);CHKERRQ(ierr);
  }
  ierr = PetscSectionViewFromOptions(s,NULL,"-section_with_sym_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}




#undef __FUNCT__
#define __FUNCT__ "dmCreateSEMSection"
PetscErrorCode dmCreateSEMSection(DM *dm, AppCtx user){

  PetscErrorCode ierr;
  PetscSection   s;


  PetscFunctionBeginUser;
    /* Create a section for SEM order k */
    {
      PetscInt       *numDof, d;
      ierr = PetscMalloc1(user.Nf*(user.dim+1), &numDof);CHKERRQ(ierr);

      PetscInt       size = 0, f;
      for (f = 0; f < user.Nf; ++f) {
        for (d = 0; d <= user.dim; ++d) numDof[f*(user.dim+1)+d] = PetscPowInt(user.k[f]-1, d)*user.Nc[f];
        size += PetscPowInt(user.k[f]+1, d)*user.Nc[f];
      }
      ierr = DMPlexCreateSection(*dm, user.dim, user.Nf, user.Nc, numDof, 0, NULL, NULL, NULL, NULL, &s);CHKERRQ(ierr);
      ierr = SetSymmetries(*dm, s, &user);CHKERRQ(ierr);
      ierr = PetscFree(numDof);CHKERRQ(ierr);
    }
    ierr = DMSetDefaultSection(*dm, s);CHKERRQ(ierr);
    ierr = PetscSectionView(s, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = DMPlexCreateSpectralClosurePermutation(*dm, PETSC_DETERMINE,s);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  PetscErrorCode    ierr;
  DM                dm;
  AppCtx	          user;
  PetscInt          Ulsz;
  Vec               Ul;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help); CHKERRQ(ierr);
    ierr = processUserOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
    // for(PetscInt i=0; i<1;i++)
    //     ierr = PetscPrintf(PETSC_COMM_SELF, "Nc %D\n", user.Nc[i]);CHKERRQ(ierr);
    ierr = createDistributedDM(PETSC_COMM_WORLD, user, &dm);CHKERRQ(ierr);
    ierr = dmCreateSEMSection(&dm, user);CHKERRQ(ierr);
    ierr = DMGetLocalVector(dm,&Ul);CHKERRQ(ierr);
    ierr = VecGetSize(Ul, &Ulsz);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "Vec Size %D\n", Ulsz);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
return 0;
}//end of main
