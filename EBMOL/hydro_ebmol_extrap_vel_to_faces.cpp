/**
 * \file hydro_ebmol_extrap_vel_to_faces.cpp
 * \addtogroup EBMOL
 * @{
 *
 */

#include <hydro_mol.H>
#include <hydro_ebmol.H>
#include <AMReX_MultiCutFab.H>


using namespace amrex;

//
// Compute upwinded FC velocities by extrapolating CC values in SPACE ONLY
// This is NOT a Godunov type extrapolation: there is NO dependence on time!
// The resulting FC velocities are computed at the CENTROID of the face.
//
void
EBMOL::ExtrapVelToFaces ( const MultiFab&  a_vel,
                          AMREX_D_DECL( MultiFab& a_umac,
                                        MultiFab& a_vmac,
                                        MultiFab& a_wmac ),
                          const Geometry&  a_geom,
                          const Vector<BCRec>& h_bcrec,
                          BCRec  const* d_bcrec)
{
    BL_PROFILE("EBMOL::ExtrapVelToFaces");

#if defined(AMREX_USE_EB) && !defined(HYDRO_NO_EB)
    AMREX_ASSERT(a_vel.hasEBFabFactory());
    auto const& fact  = dynamic_cast<EBFArrayBoxFactory const&>(a_vel.Factory());
    auto const& flags = fact.getMultiEBCellFlagFab();
    auto const& fcent = fact.getFaceCent();
    auto const& ccent = fact.getCentroid();
#endif

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(a_vel, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            AMREX_D_TERM( Box const& ubx = mfi.nodaltilebox(0);,
                          Box const& vbx = mfi.nodaltilebox(1);,
                          Box const& wbx = mfi.nodaltilebox(2););

            AMREX_D_TERM( Array4<Real> const& u = a_umac.array(mfi);,
                          Array4<Real> const& v = a_vmac.array(mfi);,
                          Array4<Real> const& w = a_wmac.array(mfi););

            Array4<Real const> const& vcc = a_vel.const_array(mfi);

#if defined(AMREX_USE_EB) && !defined(HYDRO_NO_EB)
            Box const& bx = mfi.tilebox();
            EBCellFlagFab const& flagfab = flags[mfi];
            Array4<EBCellFlag const> const& flagarr = flagfab.const_array();
            auto const typ = flagfab.getType(amrex::grow(bx,2));
            if (typ == FabType::covered)
            {
                amrex::ParallelFor(ubx, [u]
                AMREX_GPU_DEVICE (int i, int j, int k) noexcept { u(i,j,k) = 0.0; });
                amrex::ParallelFor(vbx, [v]
                AMREX_GPU_DEVICE (int i, int j, int k) noexcept { v(i,j,k) = 0.0; });
#if (AMREX_SPACEDIM==3)
                amrex::ParallelFor(wbx, [w]
                AMREX_GPU_DEVICE (int i, int j, int k) noexcept { w(i,j,k) = 0.0; });
#endif
            }
            else if (typ == FabType::singlevalued)
            {
                AMREX_D_TERM( Array4<Real const> const& fcx = fcent[0]->const_array(mfi);,
                              Array4<Real const> const& fcy = fcent[1]->const_array(mfi);,
                              Array4<Real const> const& fcz = fcent[2]->const_array(mfi););

                Array4<Real const> const& ccc = ccent.const_array(mfi);
                auto vfrac = fact.getVolFrac().const_array(mfi);

                EBMOL::ExtrapVelToFacesBox(AMREX_D_DECL(ubx,vbx,wbx),
                                           AMREX_D_DECL(u,v,w),vcc,flagarr,
                                           AMREX_D_DECL(fcx,fcy,fcz),ccc, vfrac,
                                           a_geom, h_bcrec, d_bcrec);
            }
            else
#endif
            {
                MOL::ExtrapVelToFacesBox(AMREX_D_DECL(ubx,vbx,wbx),
                                         AMREX_D_DECL(u,v,w),
                                         vcc,a_geom,h_bcrec, d_bcrec);
            }
        }
    }

}
/** @} */
