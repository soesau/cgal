// Copyright (c) 2022-2024 INRIA Sophia-Antipolis (France), GeometryFactory (France).
// All rights reserved.
//
// This file is part of CGAL (www.cgal.org).
//
// $URL$
// $Id$
// SPDX-License-Identifier: GPL-3.0-or-later OR LicenseRef-Commercial
//
// Author(s)     : Julian Stahl
//                 Mael Rouxel-Labbé

#ifndef CGAL_ISOSURFACING_3_MARCHING_CUBES_DOMAIN_3_H
#define CGAL_ISOSURFACING_3_MARCHING_CUBES_DOMAIN_3_H

#include <CGAL/license/Isosurfacing_3.h>

#include <CGAL/Isosurfacing_3/internal/Isosurfacing_domain_3.h>
#include <CGAL/Isosurfacing_3/edge_intersection_oracles_3.h>

namespace CGAL {
namespace Isosurfacing {

/**
 * \ingroup IS_Domains_grp
 *
 * \cgalModels{IsosurfacingDomain_3}
 *
 * \brief A domain that can be used with the Marching Cubes algorithm.
 *
 * \details This class is essentially wrapper around the different bricks provided by its
 * template parameters: `Partition` provides the spatial partitioning, `ValueField`
 * the values that define the isosurface. The optional template parameter
 * `EdgeIntersectionOracle` gives control over the method used to compute edge-isosurface intersection points.
 *
 * \tparam Partition must be a model of `IsosurfacingPartition_3`
 * \tparam ValueField must be a model of `IsosurfacingValueField_3`
 * \tparam EdgeIntersectionOracle must be a model of `IsosurfacingEdgeIntersectionOracle_3`
 *
 * \sa `CGAL::Isosurfacing::marching_cubes()`
 * \sa `CGAL::Isosurfacing::Dual_contouring_domain_3`
 */
template <typename Partition,
          typename ValueField,
          typename EdgeIntersectionOracle = CGAL::Isosurfacing::Linear_interpolation_edge_intersection>
class Marching_cubes_domain_3
#ifndef DOXYGEN_RUNNING
  : public internal::Isosurfacing_domain_3<Partition, ValueField, EdgeIntersectionOracle>
#endif
{
private:
  using Base = internal::Isosurfacing_domain_3<Partition, ValueField, EdgeIntersectionOracle>;

public:
  /**
   * \brief constructs a domain that can be used with the Marching Cubes algorithm.
   *
   * \param partition the space partitioning data structure
   * \param values a continuous field of scalar values, defined over the geometric span of `partition`
   * \param intersection_oracle the oracle for edge-isosurface intersection computation
   *
   * \warning the domain class keeps a reference to the `partition`, `values` and `gradients` objects.
   * As such, users must ensure that the lifetimes of these objects exceed that of the domain object.
   */
  Marching_cubes_domain_3(const Partition& partition,
                          const ValueField& values,
                          const EdgeIntersectionOracle& intersection_oracle = EdgeIntersectionOracle())
    : Base(partition, values, intersection_oracle)
  { }
};

/**
 * \ingroup IS_Domains_grp
 *
 * \brief creates a new instance of a domain that can be used with the Marching Cubes algorithm.
 *
 * \tparam Partition must be a model of `IsosurfacingPartition_3`
 * \tparam ValueField must be a model of `IsosurfacingValueField_3`
 * \tparam EdgeIntersectionOracle must be a model of `IsosurfacingEdgeIntersectionOracle_3`
 *
 * \param partition the space partitioning data structure
 * \param values a continuous field of scalar values, defined over the geometric span of `partition`
 * \param intersection_oracle the oracle for edge-isosurface intersection computation
 *
 * \warning the domain class keeps a reference to the `partition`, `values` and `gradients` objects.
 * As such, users must ensure that the lifetimes of these objects exceed that of the domain object.
 */
template <typename Partition,
          typename ValueField,
          typename EdgeIntersectionOracle = Linear_interpolation_edge_intersection>
Marching_cubes_domain_3<Partition, ValueField, EdgeIntersectionOracle>
create_marching_cubes_domain_3(const Partition& partition,
                               const ValueField& values,
                               const EdgeIntersectionOracle& intersection_oracle = EdgeIntersectionOracle())
{
  return { partition, values, intersection_oracle };
}

} // namespace Isosurfacing
} // namespace CGAL

#endif // CGAL_ISOSURFACING_3_MARCHING_CUBES_DOMAIN_3_H
