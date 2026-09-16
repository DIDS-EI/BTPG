"""Consistency checks for the VirtualHome action model.

Runs without the simulator: the action classes are loaded with stub parent
packages, so no Unity, Qt or omnigibson dependency is needed.

    python test_examples/check_vh_action_model.py

Exits non-zero if any check fails.
"""
import itertools
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LIB = os.path.join(ROOT, 'btpg', 'envs', 'virtualhome', 'exec_lib')


def _load_action_classes():
    """Import exec_lib's action classes without importing the btpg package."""
    class _Action(object):
        pass

    class _Status(object):
        RUNNING = 'running'

    def stub(name, path=None, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        if path is not None:
            m.__path__ = [path]
        sys.modules[name] = m
        return m

    stub('btpg')
    stub('btpg.behavior_tree', Status=_Status)
    stub('btpg.behavior_tree.base_nodes', Action=_Action)
    stub('btpg.envs')
    stub('btpg.envs.virtualhome')
    stub('btpg.envs.virtualhome.exec_lib', LIB)
    stub('btpg.envs.virtualhome.exec_lib._base', os.path.join(LIB, '_base'))
    stub('btpg.envs.virtualhome.exec_lib.Action', os.path.join(LIB, 'Action'))

    import importlib
    classes = {}
    for f in sorted(os.listdir(os.path.join(LIB, 'Action'))):
        if not f.endswith('.py') or f.startswith('__'):
            continue
        mod = importlib.import_module('btpg.envs.virtualhome.exec_lib.Action.' + f[:-3])
        cls = getattr(mod, f[:-3], None)
        if cls is not None:
            classes[f[:-3]] = cls
    return classes


def ground(classes):
    """All grounded actions of the expandable classes, as (class, args, info)."""
    out = []
    for name, cls in sorted(classes.items()):
        if not getattr(cls, 'can_be_expanded', False):
            continue
        n = getattr(cls, 'num_args', 0)
        args = getattr(cls, 'valid_args', [])
        combos = [()] if n == 0 else [(a,) for a in sorted(args)] if n == 1 else sorted(args)
        for c in combos:
            out.append((name, c, cls.get_info(*c)))
    return out


def symbol(atom):
    return atom.split('(')[0]


def check_no_write_only_predicate(grounded):
    """A predicate that is added but never deleted can never become false again."""
    added, deleted = set(), set()
    for _, _, info in grounded:
        added |= {symbol(a) for a in info['add']}
        deleted |= {symbol(a) for a in info['del_set']}
    bad = sorted(added - deleted)
    return bad, 'added but never deleted: %s' % (bad,)


def check_strips_wellformed(grounded):
    """add and del must be disjoint, and add must not repeat a precondition."""
    bad = []
    for name, args, info in grounded:
        pre, add, dl = set(info['pre']), set(info['add']), set(info['del_set'])
        if add & dl:
            bad.append('%s%s adds and deletes %s' % (name, args, sorted(add & dl)))
        if add & pre:
            bad.append('%s%s adds its own precondition %s' % (name, args, sorted(add & pre)))
    return bad, '\n  '.join(bad)


def check_addable_atoms_have_condition(grounded):
    """Anything an action can establish should be expressible as a goal."""
    have = {f[:-3] for f in os.listdir(os.path.join(LIB, 'Condition'))
            if f.endswith('.py') and not f.startswith('__')}
    added = set()
    for _, _, info in grounded:
        added |= {symbol(a) for a in info['add']}
    bad = sorted(added - have)
    return bad, 'no node under Condition/: %s' % (bad,)


def check_switchon_keeps_base_preconditions(classes):
    """Regression: a branch in get_info must add to `pre`, not replace it.

    `SwitchOn` used to assign `info["pre"]` again inside its `CAN_OPEN` branch,
    dropping the hand, proximity and switched-off preconditions for objects
    that are both switchable and openable.
    """
    bad = []
    for name, cls in sorted(classes.items()):
        if 'SwitchOn' not in name or not getattr(cls, 'can_be_expanded', False):
            continue
        both = sorted(set(cls.HasSwitchObjects) & set(cls.CAN_OPEN))
        for x in both:
            pre = set(cls.get_info(x)['pre'])
            syms = {symbol(a) for a in pre}
            for need in ('IsNear', 'IsSwitchedOff'):
                if need not in syms:
                    bad.append('%s(%s) lost %s' % (name, x, need))
            if not any(s.endswith('HandEmpty') for s in syms):
                bad.append('%s(%s) lost the free-hand precondition' % (name, x))
    return bad, '\n  '.join(bad)


def check_small_scene_reachable_states(classes, cap=500000):
    """Enumerate a small scene and assert no physically impossible state is reachable.

    The class-level object sets are left untouched: `valid_args` is evaluated at
    import time, so narrowing them afterwards would desynchronise `add` from
    `del_set`. Instead the grounded actions are filtered down to a few names.
    """
    scene = {'milk', 'chicken', 'kitchentable', 'desk', 'fridge', 'tv', 'self'}
    ops = []
    for _, args, info in ground(classes):
        if all(a in scene for a in args):
            ops.append((frozenset(info['pre']), frozenset(info['add']),
                        frozenset(info['del_set'])))
    init = frozenset(
        {'IsStanding(self)', 'IsLeftHandEmpty(self)', 'IsRightHandEmpty(self)',
         'IsNear(self,desk)', 'IsClose(fridge)', 'IsSwitchedOff(tv)',
         'IsOn(milk,desk)', 'IsOn(chicken,desk)'})

    seen, stack = {init}, [init]
    while stack and len(seen) < cap:
        s = stack.pop()
        for pre, add, dl in ops:
            if pre <= s:
                t = (s | add) - dl
                if t not in seen:
                    seen.add(t)
                    stack.append(t)

    bad = []
    for s in seen:
        for o in ('milk', 'chicken'):
            where = [a for a in s
                     if a.startswith(('IsOn(%s,' % o, 'IsIn(%s,' % o))
                     or a.endswith('Holding(self,%s)' % o)]
            if len(where) > 1:
                bad.append('%s is in %d places at once: %s' % (o, len(where), sorted(where)))
        for h in ('Left', 'Right'):
            if 'Is%sHandEmpty(self)' % h in s and 'Is%sHandFull(self)' % h in s:
                bad.append('the %s hand is empty and full at once' % h.lower())
        if 'IsOpen(fridge)' in s and 'IsClose(fridge)' in s:
            bad.append('fridge is open and closed at once')
        if bad:
            break
    detail = '%d actions, %d reachable states' % (len(ops), len(seen))
    for b in bad:
        detail += '\n  ' + b
    return bad, detail


def main():
    classes = _load_action_classes()
    grounded = ground(classes)
    print('loaded %d action classes, %d grounded actions'
          % (len(classes), len(grounded)))

    results = [
        ('no write-only predicate', check_no_write_only_predicate(grounded)),
        ('STRIPS well-formedness', check_strips_wellformed(grounded)),
        ('addable atoms have a Condition node',
         check_addable_atoms_have_condition(grounded)),
        ('SwitchOn keeps its base preconditions',
         check_switchon_keeps_base_preconditions(classes)),
        ('small scene has no impossible reachable state',
         check_small_scene_reachable_states(classes)),
    ]
    failed = 0
    for label, (bad, detail) in results:
        if bad:
            failed += 1
            print('FAIL  %s\n  %s' % (label, detail))
        else:
            print('ok    %s' % label)
    if failed:
        print('\n%d/%d checks failed' % (failed, len(results)))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
