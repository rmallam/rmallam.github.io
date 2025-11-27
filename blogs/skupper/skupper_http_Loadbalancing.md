In this article, We will explore how skupper helps easy it is to load balance application running on multiple kubernetes clusters

## Pre-requisites

1. Two Kubernetes/Openshift Clusters
2. Skupper Cli installed on your local

### On Terminal for Cluster1 -- Setting up Skupper 

```bash
export KUBECONFIG=~/.kube/cluster1Kube-config
oc login to cluster1
oc new-project hello-world
skupper init
skupper token create cluster1-token
```

### On a new Terminal for Cluster2 -- Setting up skupper and linking it with cluster1

```bash
export KUBECONFIG=~/.kube/cluster2Kube-config
oc login to cluster2
oc new-project hello-world
skupper init
skupper link create cluster1-token --name Cluster2-link-to-cluster1
skupper link status
```

### Install the required applications on cluster1

#### Navigate to terminal for cluster1

```bash
kubectl create deployment hello-world-frontend --image quay.io/skupper/hello-world-frontend
kubectl create deployment hello-world-backend --image quay.io/skupper/hello-world-backend
```

### Install the required applications on cluster2

#### Navigate to terminal for cluster2

```bash
kubectl create deployment hello-world-frontend --image quay.io/skupper/hello-world-frontend
kubectl create deployment hello-world-backend --image quay.io/skupper/hello-world-backend
```

### Expose a skupper Service on Cluster1 and see it being visible on Cluster2

#### On Cluster1 Terminal
```bash
# expose frontend as a openshift Service
oc expose deployment/hello-world-frontend --port 8080

# expose backend as a skupper Service
skupper expose deployment/hello-world-backend

# check skupper service status
skupper service status

# create a route for frontend service
oc expose service/hello-world-frontend

# get the route URL
oc get route hello-world-frontend -o jsonpath='{.spec.host}'
```

#### On Cluster2 Terminal
```bash
# expose frontend as a openshift Service
oc expose deployment/hello-world-frontend --port 8080

# expose backend as a skupper Service
skupper expose deployment/hello-world-backend

# check skupper service status
skupper service status

# create a route for frontend service
oc expose service/hello-world-frontend

# get the route URL
oc get route hello-world-frontend -o jsonpath='{.spec.host}'
```

### Test the Setup

#### Open a new Terminal window

```bash
export KUBECONFIG=~/.kube/cluster1Kube-config
while true; do curl `oc get route hello-world-frontend -o jsonpath='{.spec.host}'`; done
```

Observe the response coming from cluster1 backend pods

#### Open a new Terminal window

```bash
export KUBECONFIG=~/.kube/cluster2Kube-config
while true; do curl `oc get route hello-world-frontend -o jsonpath='{.spec.host}'`; done
```

Observe the response coming from cluster2 backend pods

## Failover test

### On cluster2 terminal

```bash
# unexpose backend service from skupper
skupper unexpose deployment/hello-world-backend
```

We have now unexposed the backend service from skupper on cluster2, but it is still exposed from cluster1 and is synced back to cluster2. So The backend response should now be coming from cluster1 instead of cluster2. Check the response in the terminal window where cluster2 curl is running.