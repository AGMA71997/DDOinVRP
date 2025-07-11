from collections import deque
# from functools import lru_cache
from utils import *
from instance_generator import Instance_Generator
import time
import random


class Label:
    """A label describes a path from the depot to a customer and the resources
       used in this path.

       Labels are associated with a customer and are used to identify each
       feasible state in which that customer can be reached.

       A dominance relation between labels is needed (see 'dominates' method for
       more information) so only 'best' labels can be kept on a node resulting
       in fewer paths to be explored.
    """

    def __init__(self, customer, cost, load, time, customer_labels, prev=None):
        self.customer = customer
        self.cost = cost
        self.load = load
        self.time = time
        self.unreachable_cs = set()
        self.prev = prev
        self.customer_labels = customer_labels
        self.dominated = False

    def __repr__(self):
        return f"{(self.customer, self.cost, self.load, self.time)}"

    # @property
    # @lru_cache(maxsize=1)
    def path(self):
        """Returns the path described by this label."""

        label = self
        path = []
        while label.prev:
            path.append(label.customer)
            label = label.prev
        path.append(label.customer)
        return list(reversed(path))

    def dominates(self, label):
        """A label is called 'dominant' when compared to another label, if
           it uses less resources than the other label, meaning that all of
           its resources must be less than or equal to the other's, but there
           must be at least one resource that is different.

           Since two perfectly equals labels describe the same path, it is
           possible to throw away one of them by modifying the above definition
           and letting l1 dominate l2 even if they have the same resources.

           Arguments:
               label: the label to compare with
           Returns:
               True if this label dominates 'label', False otherwise
        """

        return (self.cost <= label.cost and self.load <= label.load
                and self.time <= label.time)

    def is_dominated(self):
        """Returns True if this label is dominated by any of the labels
           associated with the same customer, False otherwise."""
        for label in self.customer_labels[self.customer]:
            if label.dominates(self):
                return True
        return False

    def filter_dominated(self):
        """Removes labels dominated by this label on its customer."""

        labels = []
        for label in self.customer_labels[self.customer]:
            if self.dominates(label):
                # label can be already in the 'to_be_extended' queue
                # so we need to signal that this label is no more extendable
                label.dominated = True
            else:
                labels.append(label)
        self.customer_labels[self.customer] = labels


class ESP_Label(Label):
    """Extension of base Label used to describe elementary paths only.
       To do that the unreachable customers set contains also the visited
       customers and it is used as a resource, meaning that the dominance
       relation is also extended."""

    def __init__(self, *args):
        super().__init__(*args)
        self.unreachable_cs.add(self.customer)

    def dominates(self, label):
        # Note that having the unreachable customers set as a resource means
        # that a label uses less of this resource if it possesses a subset of
        # the other's unreachable customers set.
        return (super().dominates(label)
                and self.unreachable_cs.issubset(label.unreachable_cs))


class ESPPRC:
    """The Elementary Shortest Path Problem with Resource Constraints
       instance class.

       It stores instance data and is able to solve the ESPPRC problem with an
       exact dynamic programming approach.
       It uses a dual variable array so it can be used in a column generation
       approach for vehicle routing problems.
    """

    def __init__(self, vehicle_capacity, demands, time_windows, service_times, N,
                 time_matrix, prices):

        self.capacity = vehicle_capacity
        self.demands = demands
        self.time_windows = time_windows
        self.service_times = service_times
        self.n_customers = N
        self.times = time_matrix

        self.costs = prices

        self.label_cls = ESP_Label
        customer_labels = {}
        for customer in range(self.n_customers + 1):
            customer_labels[customer] = []
        self.customer_labels = customer_labels

    def solve(self):

        to_be_extended = deque([self.depot_label()])
        while to_be_extended:
            from_label = to_be_extended.popleft()
            # if a label becomes dominated after being pushed in the queue,
            # label.dominated becomes true and it can be skipped
            if from_label.dominated:
                continue

            to_labels = self.feasible_labels_from(from_label)
            for to_label in to_labels:
                to_cus = to_label.customer
                if to_cus != 0:
                    to_label.unreachable_cs.update(from_label.unreachable_cs)
                    if to_label.is_dominated():
                        continue
                    to_label.filter_dominated()
                    # if len(self.customer_labels[to_cus]) < 20: ###################
                    to_be_extended.append(to_label)
                self.customer_labels[to_cus].append(to_label)

        labels = sorted(self.customer_labels[0], key=lambda x: x.cost)

        final_labels = [label for label in labels if label.cost < -0.001]
        return final_labels

    def depot_label(self):
        """Returns the algorithm starting label. It has no resources and its
           path can return to the depot."""
        label = self.label_cls(0, 0, 0, 0, self.customer_labels)
        label.unreachable_cs.clear()
        return label

    def feasible_labels_from(self, from_label):
        """Arguments:
               from_label: the label that is going to be extended.
           Returns:
               A list of feasible labels that extends 'from_label'.
           Note: 'from_label' unreachable set is updated in the process.
        """

        to_labels = []
        for to_cus in range(self.n_customers + 1):
            if to_cus in from_label.unreachable_cs or to_cus == from_label.customer:
                continue
            to_label = self.extended_label(from_label, to_cus)
            if not to_label:
                from_label.unreachable_cs.add(to_cus)
            else:
                to_labels.append(to_label)
        return to_labels

    def extended_label(self, from_label, to_cus):
        """Returns a new label that extends 'from_label' and goes to 'to_cus'.

           Arguments:
               from_label: the label to start from
               to_cus: the customer to reach
           Returns:
               A new label with updated resources or None if some resource
               exceeds its limits.
        """

        load = from_label.load + self.demands[to_cus]
        if load > self.capacity:
            return

        from_cus = from_label.customer
        if self.costs[from_cus, to_cus] == math.inf:
            return
        time = max(from_label.time + self.service_times[from_cus]
                   + self.times[from_cus, to_cus],
                   self.time_windows[to_cus, 0])
        if time > self.time_windows[to_cus, 1]:
            return

        cost = from_label.cost + self.costs[from_cus, to_cus]
        # unreachable customers update is delayed since from_label needs to
        # visit every customer before knowing its own set
        return self.label_cls(to_cus, cost, load, time, self.customer_labels, from_label)


#######################

class SSR_Label(Label):
    """State space relaxed version of ESP Label. The relaxation is accomplished
       by not using the unreachable customer set as a resource, replacing it
       with the number of visited customers. By doing so the dominated labels
       grow in number and less paths are explored reducing the execution time,
       but it is possible to have cycles in the path found."""

    def __init__(self, *args):
        super().__init__(*args)
        self.n_visited = 0

    def dominates(self, label):
        return super().dominates(label) and self.n_visited <= label.n_visited


class SSR_SPPRC(ESPPRC):
    """State space relaxation SPPRC algorithm. See SSR label for details."""

    def __init__(self, *args):
        super().__init__(*args)
        self.label_cls = SSR_Label

    def extended_label(self, from_label, to_cus):
        label = super().extended_label(from_label, to_cus)
        if not label:
            return

        n_visited = from_label.n_visited + 1
        if n_visited > self.n_customers:
            return

        label.n_visited = n_visited
        return label


##################################################
def find_repeated(items):
    """Arguments:
           items: the items to be searched for repeated elements.
       Returns:
           A set of elements that are repeated in 'items'."""

    seen, seen_twice = set(), set()
    seen_twice_add = seen_twice.add
    seen_add = seen.add
    for item in items:
        if item in seen:
            seen_twice_add(item)
        else:
            seen_add(item)
    return seen_twice


class DSSR_Label(SSR_Label):
    """Decremental state space relaxed version of ESP Label. Same as SSR Label
       but with an added resource, the critical customers visited.
       Critical customers are customers that the algorithm previously find as
       repeated in the SSR SPPRC execution.
       The new resource doesn't allow to visit critical customers twice (but
       only them), so it prevents the same cycle to appear again."""

    def __init__(self, *args):
        super().__init__(*args)
        self.critical_visited = set()

    def dominates(self, label):
        return (super().dominates(label)
                and self.critical_visited.issubset(label.critical_visited))


class DSSR_ESPPRC(SSR_SPPRC):
    """Decremental state space relaxation ESPPRC algorithm. It starts by solving
       the SPPRC using DSSR labels, which are basically SSR labels, then if it
       finds a cycle (repeated customers in the resulting path) it keeps tracks
       of those customers (called critical customers) and restart the algorithm.
       The DSSR labels used (see the respective documentation) prevent critical
       customers to be visited twice so the process is repeated until an acyclic
       path is returned."""

    def __init__(self, *args):
        super().__init__(*args)
        self.label_cls = DSSR_Label
        self.critical_cs = set()

    def solve(self):
        labels = super().solve()

        def acyclic_labels():
            for label in labels:
                repeated = find_repeated(label.path()[:-1])
                if repeated:
                    self.critical_cs.update(repeated)
                else:
                    yield label
                assert label.path()[-1] == 0 and label.path()[0] == 0 and label.cost < 0

        final_labels = list(acyclic_labels())
        # print(len(final_labels))
        # final_labels = final_labels[:min(200, len(final_labels))] ###################
        return final_labels

    def extended_label(self, from_label, to_cus):
        label = super().extended_label(from_label, to_cus)
        if not label or to_cus in from_label.critical_visited:
            return

        label.critical_visited.update(from_label.critical_visited)
        if to_cus in self.critical_cs:
            label.critical_visited.add(to_cus)
        return label


def main():
    random.seed(10)
    numpy.random.seed(15)
    torch.manual_seed(20)

    for exp in range(10):
        num_customers = 100
        VRP_instance = Instance_Generator(N=num_customers)
        capacity = VRP_instance.vehicle_capacity
        demands = VRP_instance.demands
        time_windows = VRP_instance.time_windows
        service_times = VRP_instance.service_times
        n_customers = VRP_instance.N
        times = VRP_instance.time_matrix
        duals = create_duals(1, n_customers,
                             torch.tensor(times.reshape(1, n_customers + 1, n_customers + 1)))
        duals = duals.tolist()[0]
        prices = create_price(times, duals) * -1
        algo = DSSR_ESPPRC(capacity, demands, time_windows, service_times, n_customers,
                           times, prices)

        time1 = time.time()
        opt_labels = algo.solve()
        print("The total number of labels is: " + str(len(opt_labels)))
        print("The total run time is: " + str(time.time() - time1))
        print(opt_labels[0].cost)
        print(opt_labels[0].path())
        for label in opt_labels:
            if label.customer != 0:
                print(label)
                print("Mistake")
        print("-------------------------")


if __name__ == "__main__":
    main()
