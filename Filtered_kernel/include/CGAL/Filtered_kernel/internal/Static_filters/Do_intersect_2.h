// Copyright (c) 2008 ETH Zurich (Switzerland)
// Copyright (c) 2008-2009 INRIA Sophia-Antipolis (France)
// Copyright (c) 2017 GeometryFactory Sarl (France)
// All rights reserved.
//
// This file is part of CGAL (www.cgal.org)
//
// $URL$
// $Id$
// SPDX-License-Identifier: LGPL-3.0-or-later OR LicenseRef-Commercial
//
//
// Author(s)     : Andreas Fabri, Laurent Rineau


#ifndef CGAL_INTERNAL_STATIC_FILTERS_DO_INTERSECT_2_H
#define CGAL_INTERNAL_STATIC_FILTERS_DO_INTERSECT_2_H

#include <iostream>

namespace CGAL {

namespace internal {

namespace Static_filters_predicates {

template < typename K_base, typename SFK >
class Do_intersect_2
  : public K_base::Do_intersect_2
{
  typedef typename K_base::Boolean   Boolean;
  typedef typename K_base::Point_2   Point_2;
  typedef typename K_base::Segment_2 Segment_2;
  typedef typename K_base::Circle_2  Circle_2;

  typedef typename K_base::Do_intersect_2 Base;

public:
  using Base::operator();

  // The internal::do_intersect(..) function
  // only performs orientation tests on the vertices
  // of the segment
  // By calling the do_intersect function with
  // the  statically filtered kernel we avoid
  // that doubles are put into Interval_nt
  // to get taken out again with fit_in_double
  Boolean
  operator()(const Segment_2 &s, const Segment_2& t) const
  {
    return Intersections::internal::do_intersect(s,t, SFK());
  }

  Boolean
  operator()(const Point_2 &p, const Segment_2& t) const
  {
    return Intersections::internal::do_intersect(p,t, SFK());
  }

  Boolean
  operator()(const Segment_2& t, const Point_2 &p) const
  {
    return Intersections::internal::do_intersect(p,t, SFK());
  }

  // The parameter overestimate is used to avoid a filter failure in AABB_tree::closest_point()
  Boolean
    operator()(const Circle_2& s, const Bbox_2& b, bool overestimate = false) const
  {
    CGAL_BRANCH_PROFILER_3(std::string("semi-static failures/attempts/calls to   : ") +
      std::string(CGAL_PRETTY_FUNCTION), tmp);

    Get_approx<Point_2> get_approx; // Identity functor for all points
    const Point_2& c = s.center();

    double scx, scy, ssr;
    double bxmin = b.xmin(), bymin = b.ymin(),
      bxmax = b.xmax(), bymax = b.ymax();

    if (fit_in_double(get_approx(c).x(), scx) &&
      fit_in_double(get_approx(c).y(), scy) &&
      fit_in_double(s.squared_radius(), ssr))
    {
      CGAL_BRANCH_PROFILER_BRANCH_1(tmp);

      if ((ssr < 1.11261183279326254436e-293) || (ssr > 2.80889552322236673473e+306)) {
        CGAL_BRANCH_PROFILER_BRANCH_2(tmp);
        return Base::operator()(s, b);
      }
      double distance = 0;
      double max1 = 0;
      double double_tmp_result = 0;
      double eps = 0;
      if (scx < bxmin)
      {
        double bxmin_scx = bxmin - scx;
        max1 = bxmin_scx;

        distance = square(bxmin_scx);
        double_tmp_result = (distance - ssr);

        if ((max1 < 3.33558365626356687717e-147) || (max1 > 1.67597599124282407923e+153)) {
          if (overestimate) {
            return true;
          }
          else {
            CGAL_BRANCH_PROFILER_BRANCH_2(tmp);
            return Base::operator()(s, b);
          }
        }

        eps = 1.99986535548615598560e-15 * (std::max)(ssr, square(max1));

        if (double_tmp_result > eps) {
          return false;
        }
      }
      else if (scx > bxmax)
      {
        double scx_bxmax = scx - bxmax;
        max1 = scx_bxmax;

        distance = square(scx_bxmax);
        double_tmp_result = (distance - ssr);

        if ((max1 < 3.33558365626356687717e-147) || (max1 > 1.67597599124282407923e+153)) {
          if (overestimate) {
            return true;
          }
          else {
            CGAL_BRANCH_PROFILER_BRANCH_2(tmp);
            return Base::operator()(s, b);
          }
        }

        eps = 1.99986535548615598560e-15 * (std::max)(ssr, square(max1));

        if (double_tmp_result > eps) {
          return false;
        }
      }


      if (scy < bymin)
      {
        double bymin_scy = bymin - scy;
        if (max1 < bymin_scy) {
          max1 = bymin_scy;
        }

        distance += square(bymin_scy);
        double_tmp_result = (distance - ssr);

        if ((max1 < 3.33558365626356687717e-147) || ((max1 > 1.67597599124282407923e+153))) {
          if (overestimate) {
            return true;
          }
          else {
            CGAL_BRANCH_PROFILER_BRANCH_2(tmp);
            return Base::operator()(s, b);
          }
        }

        eps = 1.99986535548615598560e-15 * (std::max)(ssr, square(max1));

        if (double_tmp_result > eps) {
          return false;
        }
      }
      else if (scy > bymax)
      {
        double scy_bymax = scy - bymax;
        if (max1 < scy_bymax) {
          max1 = scy_bymax;
        }
        distance += square(scy_bymax);
        double_tmp_result = (distance - ssr);

        if (((max1 < 3.33558365626356687717e-147)) || ((max1 > 1.67597599124282407923e+153))) {
          if (overestimate) {
            return true;
          }
          else {
            CGAL_BRANCH_PROFILER_BRANCH_2(tmp);
            return Base::operator()(s, b);
          }
        }

        eps = 1.99986535548615598560e-15 * (std::max)(ssr, square(max1));

        if (double_tmp_result > eps) {
          return false;
        }
      }

      // double_tmp_result and eps were growing all the time
      // no need to test for > eps as done earlier in at least one case
      if (double_tmp_result < -eps) {
        return true;
      }
      else {
        if (overestimate) {
          return true;
        }
        CGAL_BRANCH_PROFILER_BRANCH_2(tmp);
        return Base::operator()(s, b);
      }

      CGAL_BRANCH_PROFILER_BRANCH_2(tmp);
    }
    return Base::operator()(s, b);
  }

};
} // Static_filters_predicates
} // internal
} // CGAL
#endif
