/**
 * \file hydro_ebgodunov_K.H
 *
 * \addtogroup EBGodunov
 *  @{
 */

#ifndef HYDRO_EBGODUNOV_K_H
#define HYDRO_EBGODUNOV_K_H

#include <AMReX_Gpu.H>
#include <AMReX_BCRec.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_Array.H>
#include <iomanip>
#include <hydro_constants.H>

/* This header file contains the inlined __host__ __device__ functions required
   for setting BCs for computing the advective velocity and transverse terms in
   3D EBGodunov ExtrapVel.
*/

namespace EBGodunovBC {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void SetXBCs (const int i, const int j, const int k, const int n,
              const amrex::Array4<const amrex::Real> &s,
              amrex::Real &lo,
              amrex::Real &hi,
              const int bclo, const int bchi,
              const int domlo, const int domhi,
              const bool is_velocity )
{
    using namespace amrex;


    // Low X
    if (i <= domlo)
    {
        if (bclo==BCType::ext_dir)
        {
            lo = s(domlo-1,j,k,n);
            // For turbulent inflow, there are times when the inflow face
            // may have a predicted outflowing velocity. Here, we preserve
            // the normal component of the Dirichlet BC, but allow the
            // tangential components to transport values from the interior.
            if( n == XVEL && is_velocity ) hi=lo;
        }
        else if (bclo == BCType::foextrap || bclo == BCType::hoextrap)
        {
            lo = hi;
        } else if (bclo == BCType::reflect_even)
        {
            lo = hi;

            if ( i<domlo)
                Abort("EBGodunovBC::SetXBCs not yet fully implemented for reflect_even BC");
            //
            // This is potentially tricky because the code re-uses some of the
            // space holding the holding the states passed in here; it puts the
            // upwinded intermediate edge state in Imx (which supplies hi)
            // With GPU, I think it's undertermined which Imx(i+..) you'll get here,
            // could be from PLM, could be the upwinded edgestate. Does this matter?
            // lo = hi_arr(2*domlo-i  ,j,k,n);
            // hi = lo_arr(2*domlo-i-1,j,k,n);
        }
        else if (bclo == BCType::reflect_odd)
        {
            if ( i==domlo ) {
                hi = 0.;
                lo = 0.;
            } else {
                Abort("EBGodunovBC::SetXBCs not yet fully implemented for reflect_odd BC");
            //     lo = -hi_arr(2*domlo-i  ,j,k,n);
            //     hi = -lo_arr(2*domlo-i-1,j,k,n);
            }
        }
    }
    // High X
    else if (i > domhi)
    {
        if (bchi==BCType::ext_dir)
        {
            hi = s(domhi+1,j,k,n) ;
            if( n ==XVEL && is_velocity ) lo=hi;
        }
        else if (bchi == BCType::foextrap || bchi == BCType::hoextrap )
        {
            hi = lo;
        }
        else if (bchi == BCType::reflect_even)
        {
            hi = lo;

            if ( i>domhi+1 )
                Abort("EBGodunovBC::SetXBCs not yet fully implemented for reflect_even BC");
            // hi = lo_arr(2*(domhi+1)-i-1,j,k,n);
            // lo = hi_arr(2*(domhi+1)-i  ,j,k,n);
        }
        else if (bchi == BCType::reflect_odd)
        {
            if ( i==domhi+1 ) {
                hi = 0.;
                lo = 0.;
            } else {
                Abort("EBGodunovBC::SetXBCs not yet fully implemented for reflect_odd BC");
            //     hi = -lo_arr(2*(domhi+1)-i-1,j,k,n);
            //     lo = -hi_arr(2*(domhi+1)-i  ,j,k,n);
            }
        }
    }
    else
        return;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void SetYBCs (const int i, const int j, const int k, const int n,
              const amrex::Array4<const amrex::Real> &s,
              amrex::Real &lo,
              amrex::Real &hi,
              const int bclo, const int bchi,
              const int domlo, const int domhi,
              const bool is_velocity )
{
    using namespace amrex;


    // Low Y
    if (j <= domlo)
    {
        if (bclo==BCType::ext_dir)
        {
            lo = s(i,domlo-1,k,n);
            if ( n == YVEL && is_velocity ) hi = lo;
        }
        else if (bclo == BCType::foextrap || bclo == BCType::hoextrap)
        {
            lo = hi;
        }
        else if (bclo == BCType::reflect_even)
        {
            lo = hi;

            if  ( j<domlo )
                Abort("EBGodunovBC::SetYBCs not yet fully implemented for reflect_even BC");
            // lo = hi_arr(i,2*domlo-j  ,k,n);
            // hi = lo_arr(i,2*domlo-j-1,k,n);
        }
        else if(bclo == BCType::reflect_odd)
        {
            if ( j==domlo ) {
                hi = 0.;
                lo = 0.;
            } else {
                Abort("EBGodunovBC::SetYBCs not yet fully implemented for reflect_odd BC");
            //     lo = -hi_arr(i,2*domlo-j  ,k,n);
            //     hi = -lo_arr(i,2*domlo-j-1,k,n);
            }
        }
    }
    // High Y
    else if (j > domhi)
    {
        if (bchi==BCType::ext_dir)
        {
            hi = s(i,domhi+1,k,n);
            if( n == YVEL && is_velocity ) lo = hi ;
        }
        else if (bchi == BCType::foextrap || bchi == BCType::hoextrap)
        {
            hi = lo;
        }
        else if (bchi == BCType::reflect_even)
        {
            hi = lo;

            if ( j>domhi+1)
                Abort("EBGodunovBC::SetYBCs not yet fully implemented for reflect_even BC");
            // hi = lo_arr(i,2*(domhi+1)-j-1,k,n);
            // lo = hi_arr(i,2*(domhi+1)-j  ,k,n);
        }
        else if (bchi == BCType::reflect_odd)
        {
            if ( j==domhi+1 ) {
                hi = 0.;
                lo = 0.;
            } else {
                Abort("EBGodunovBC::SetYBCs not yet fully implemented for reflect_odd BC");
            //     hi = -lo_arr(i,2*(domhi+1)-j-1,k,n);
            //     lo = -hi_arr(i,2*(domhi+1)-j  ,k,n);
            }
        }
    }
    else
        return;
}

#if (AMREX_SPACEDIM==3)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void SetZBCs(const int i, const int j, const int k, const int n,
             const amrex::Array4<const amrex::Real> &s,
             amrex::Real &lo,
             amrex::Real &hi,
             const int bclo, const int bchi,
             const int domlo, const int domhi,
             const bool is_velocity)
{
    using namespace amrex;


    // Low Z
    if (k <= domlo)
    {
        if (bclo==BCType::ext_dir)
        {
            lo =s(i,j,domlo-1,n);
            if ( n == ZVEL && is_velocity ) hi = lo;
        }
        else if (bclo == BCType::foextrap || bclo == BCType::hoextrap)
        {
            lo = hi;
        }
        else if (bclo == BCType::reflect_even)
        {
            Abort("EBGodunovBC::SetZBCs not yet implemented for reflect_even BC");
            // lo = hi_arr(i,j,2*domlo-k  ,n);
            // hi = lo_arr(i,j,2*domlo-k-1,n);
        }
        else if(bclo == BCType::reflect_odd)
        {
            Abort("EBGodunovBC::SetZBCs not yet implemented for reflect_odd BC");
            // if ( k==domlo ) {
            //     hi = 0.;
            //     lo = 0.;
            // } else {
            //     lo = -hi_arr(i,j,2*domlo-k  ,n);
            //     hi = -lo_arr(i,j,2*domlo-k-1,n);
            // }
        }
    }
    // High Z
    else if (k > domhi)
    {
        if (bchi==BCType::ext_dir)
        {
            hi = s(i,j,domhi+1,n);
            if ( n == ZVEL && is_velocity ) lo = hi ;
        }
        else if (bchi == BCType::foextrap || bchi == BCType::hoextrap)
        {
            hi = lo;
        }
        else if (bchi == BCType::reflect_even)
        {
            Abort("EBGodunovBC::SetZBCs not yet implemented for reflect_even BC");
            // hi = lo_arr(i,j,2*(domhi+1)-k-1,n);
            // lo = hi_arr(i,j,2*(domhi+1)-k  ,n);
        }
        else if (bchi == BCType::reflect_odd)
        {
            Abort("EBGodunovBC::SetZBCs not yet implemented for reflect_odd BC");
            // if ( k==domhi+1 ) {
            //     hi = 0.;
            //     lo = 0.;
            // } else {
            //     hi = -lo_arr(i,j,2*(domhi+1)-k-1,n);
            //     lo = -hi_arr(i,j,2*(domhi+1)-k  ,n);
            // }
        }
    }
    else
        return;
}
#endif
}
#endif
/** @} */